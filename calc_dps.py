import random
from datetime import timedelta

from osrsbox import items_api, monsters_api
import numpy as np
import matplotlib.pyplot as plt

def monster_complete(m):
    return m.defence_level >= 1 and m.hitpoints >= 1

api_monsters = monsters_api.load()
p2p_monsters = [m for m in api_monsters if monster_complete(m)]
f2p_monsters = [m for m in api_monsters if monster_complete(m) and not m.members]
monster_dict = {}

for m in p2p_monsters:
    if (m.name.lower(), m.combat_level) not in monster_dict:
        monster_dict[(m.name.lower(), m.combat_level)] = m

all_monsters = list(monster_dict.values())


def valid_equipment(i):
    return (not i.quest_item and i.equipable_by_player
            and not i.placeholder and i.equipment)


api_items = items_api.load()
all_equipment = [i for i in api_items if valid_equipment(i)]
p2p_weapons = [i for i in all_equipment if i.weapon]
f2p_weapons = [i for i in p2p_weapons if (not i.members) and i.tradeable]

weapon_dict = {}
for w in p2p_weapons:
    weapon_dict[w.name.lower()] = w

all_weapons = list(weapon_dict.values())

SLOTS = ['weapon', 'head', 'body', 'legs', 'feet', 'hands', 'cape', 'neck',
         'ring', 'shield']

STATS = ['attack_stab', 'attack_slash', 'attack_crush', 'attack_magic',
         'attack_ranged', 'defence_stab', 'defence_slash', 'defence_crush',
         'defence_magic', 'defence_ranged', 'melee_strength',
         'ranged_strength', 'magic_damage', 'prayer']

equipment_dict = {s: {} for s in SLOTS}

for i in all_equipment:
    if i.equipment.slot in SLOTS:
        equipment_dict[i.equipment.slot][i.name.lower()] = i

cache = {}


class AttackStyle(object):
    SLASH = 'slash'
    STAB = 'stab'
    CRUSH = 'crush'
    MAGIC = 'magic'
    RANGED = 'ranged'


class NoWeapon(object):
    def __init__(self):
        self.attack_speed = 4
        self.weapon_type = 'unarmed'
        self.stances = [
            {'combat_style': 'punch', 'attack_type': 'crush', 'attack_style':
             'accurate', 'experience': 'attack', 'boosts': None},
            {'combat_style': 'kick', 'attack_type': 'crush', 'attack_style':
             'aggressive', 'experience': 'strength', 'boosts': None},
            {'combat_style': 'block', 'attack_type': 'crush', 'attack_style':
             'defensive', 'experience': 'defence', 'boosts': None}
        ]


class NoEquip(object):
    def __init__(self, slot):
        self.attack_slash = 0
        self.attack_stab = 0
        self.attack_crush = 0
        self.attack_ranged = 0
        self.attack_magic = 0

        self.defence_slash = 0
        self.defence_stab = 0
        self.defence_crush = 0
        self.defence_ranged = 0
        self.defence_magic = 0

        self.melee_strength = 0
        self.ranged_strength = 0
        self.magic_damage = 0

        self.prayer = 0
        self.slot = slot
        self.requirements = {}


class NoItem(object):
    def __init__(self, slot):
        self.equipment = NoEquip(slot)
        if slot == 'weapon':
            self.weapon = NoWeapon()


class Equipment(object):
    def __init__(self, stance, **kwargs):
        """
        stance: int, index into the weapon's stances array
        """
        for s in SLOTS:
            item = kwargs.get(s)

            if not item:
                item = NoItem(s)
            elif type(item) == str:
                if s == 'weapon':
                    item = weapon_dict[item]
                else:
                    item = equipment_dict[s][item]

            setattr(self, s, item)

        self.stance = self.weapon.weapon.stances[stance]

    def __str__(self):
        slots = []
        stats = {s: 0 for s in STATS}

        for s in SLOTS:
            item = getattr(self, s)
            slots.append('%s: %s' % (s, item.name))
            for st in STATS:
                stats[st] += getattr(item.equipment, st)


        return ", ".join(slots) + '\n' + ', '.join(['%s: %s' % i for i in
                                                    stats.items()])


class Player(object):
    def __init__(self, attack=1, strength=1, defence=1, ranged=1, magic=1,
                 prayer=1, equipment=None):
        self.attack = attack
        self.strength = strength
        self.defence = defence
        self.ranged = ranged
        self.magic = magic
        self.prayer = prayer

        if equipment is None:
            equipment = Equipment(0)

        self.equip = equipment

    def get_attack_roll(self):
        atk_bonus = 0
        if self.equip.stance['attack_style'] == 'accurate':
            atk_bonus = 3
        elif self.equip.stance['attack_style'] == 'controlled':
            atk_bonus = 1

        # Calculate the player's max attack roll
        effective_atk = self.attack + atk_bonus + 8
        atype = self.equip.stance['attack_type']
        equip_atk = 0
        for s in SLOTS:
            equip = getattr(self.equip, s)
            equip_atk += getattr(equip.equipment, "attack_" + atype)

        return effective_atk * (equip_atk + 64)

    def get_max_hit(self):
        str_bonus = 0
        if self.equip.stance['attack_style'] == 'controlled':
            str_bonus = 1
        elif self.equip.stance['attack_style'] == 'aggressive':
            str_bonus = 3

        # Calculate the player's max hit
        effective_str = self.strength + str_bonus + 8
        equip_str = 0
        for s in SLOTS:
            equip = getattr(self.equip, s)
            equip_str += equip.equipment.melee_strength

        return int(0.5 + effective_str * (equip_str + 64) / 640)

    def get_defence_roll(self, attack_type):
        def_bonus = 0
        if self.equip.stance['attack_style'] == 'controlled':
            def_bonus = 1
        elif self.equip.stance['attack_style'] == 'defensive':
            def_bonus = 3

        effective_def = self.defence + def_bonus + 8

        equip_def = 0
        for s in SLOTS:
            equip = getattr(self.equip, s)

            # This workaround is here because some wiki entries only have
            # "melee" for attack style.
            if attack_type == 'melee':
                mdef = 0
                for t in ['slash', 'stab', 'crush']:
                    mdef += getattr(equip.equipment, "defence_" + t)
                equip_def += mdef / 3
            else:
                equip_def += getattr(equip.equipment, "defence_" + attack_type)

        return effective_def * (equip_def + 64)


class Encounter(object):
    def __init__(self, player, monster):
        self.player = player
        self.monster = monster


def get_atk_stats(player, enemy):
    max_hit = player.get_max_hit()
    atk_roll = player.get_attack_roll()

    # Calculate the enemy's max defence roll
    enemy_equip_bonus = getattr(enemy, "defence_" + player.equip.stance['attack_type'])
    def_roll =  (enemy.defence_level + 9) * (enemy_equip_bonus + 64)

    # Calculate the chance of a hit (not a splash)
    if atk_roll > def_roll:
        hit_chance = 1 - def_roll / (2. * atk_roll)
    else:
        hit_chance = atk_roll / (3. * def_roll)

    print('max hit: %d, hit chance: %.3f' % (max_hit, hit_chance))

    return hit_chance, max_hit


def get_def_stats(player, enemy):
    for t in enemy.attack_type:
        if t == 'typeless':
            continue
        elif t == 'curse':
            continue
        elif t == 'dragonfire':
            continue
        elif t == 'ranged':
            continue
        elif t == 'magic':
            continue

        if t in ['melee', 'slash', 'stab', 'crush']:
            # Calculate the enemy's max hit
            max_hit = int(0.5 + (enemy.strength_level + 9) * (enemy.melee_strength + 64) / 640)

            # Calculate the enemy's max attack roll
            equip_bonus = getattr(enemy, "attack_" + t, 0) + enemy.attack_accuracy
            atk_roll = (enemy.attack_level + 9) * (equip_bonus + 64)

            # player's max defence roll
            def_roll = player.get_defence_roll(t)

        # Calculate the chance of a hit (not a splash)
        if atk_roll > def_roll:
            hit_chance = 1 - def_roll / (2. * atk_roll)
        else:
            hit_chance = atk_roll / (3. * def_roll)

        print('%s: max hit: %d, hit chance: %.3f' % (t, max_hit, hit_chance))

    return hit_chance, max_hit


def find_bis_armor(player, attack_style):
    aset = {s: None for s in SLOTS}
    for slot, items in all_armor.items():
        eligible = []
        for i in items:
            if not i.equipment.requirements:
                pass
            for stat in ['defence', 'attack', 'strength', 'ranged', 'magic',
                         'prayer']:
                pass
        attr = 'defence_' + attack_style
        aset[slot] = max(items, key=lambda i: getattr(i.equipment, attr))

    return Equipment(**aset)


"""
Calculate the expected number of hits to kill an enemy, given hit chance, max
hit, and the enemy's HP.
"""
def expected_htk(hit_chance, max_hit, hp):
    ehtk = []

    for i in range(1, hp + 1):
        prevs = sum(ehtk[max(i - max_hit - 1, 0):])
        res = (1 / hit_chance) + (1. / max_hit) * prevs
        ehtk.append(res)

    return ehtk[-1]


def convolve(d1, d2, cap=None):
    if cap is not None:
        res_len = min(cap, len(d1) + len(d2) - 1)
    else:
        res_len = len(d1) + len(d2) - 1

    results = np.zeros(res_len)
    for i in range(len(d1)):
        for j in range(len(d2)):
            rix = min(i + j, len(results) - 1)
            results[rix] += d1[i] * d2[j]

    return results


def simulate_htk(hit_chance, max_hit, hp):
    probs = [1 - hit_chance] + [hit_chance / max_hit] * max_hit
    n_hit_dist = [probs]
    last = probs
    res = [0]
    while len(last) <= hp or sum(res) < 0.9999:
        n_hit_dist.append(convolve(last[:hp], probs, hp + 1))
        last = n_hit_dist[-1]
        if len(last) <= hp:
            res.append(0)
        else:
            res.append(last[hp])

    plt.bar(range(len(res)), res)
    plt.show()

    return res


def simulate_damage(hit_chance, max_hit, hit_probs):
    probs = [1 - hit_chance] + [hit_chance / max_hit] * max_hit
    n_hit_dist = [probs]
    for i in range(len(hit_probs) - 1):
        n_hit_dist.append(convolve(n_hit_dist[-1], probs))

    print(sum(hit_probs))
    print(len(n_hit_dist[-1]))

    res = np.zeros(len(n_hit_dist[-1]))
    for i, p in enumerate(hit_probs):
        res += p * np.append(n_hit_dist[i],
                             np.zeros(len(res) - len(n_hit_dist[i])))

    mean = 0
    for i in range(len(res)):
        mean += i * res[i]

    cdf = [sum(res[:i+1]) for i in range(len(res))]
    med = next(i for i, x in enumerate(cdf) if x > 0.5)
    low_bound = next(i for i, x in enumerate(cdf) if x > 0.05)
    hi_bound = next(i for i, x in enumerate(cdf) if x > 0.95)
    hi_hi_bound = next(i for i, x in enumerate(cdf) if x > 0.9999)

    print(cdf)

    print('%d < %d < %d' % (low_bound, med, hi_bound))

    res = res[:hi_hi_bound]
    plt.bar(range(len(res)), res)
    plt.show()

    return mean


KILL_DELAY = 3
BANK_DELAY = 150

"""
Compute expected time to kill a enemy given attack level, strength level,
weapon, and weapon stance.
"""
def expected_ttk(player, enemy):
    #tup = (attack, strength, enemy.id, weapon.id, stance['combat_style'])
    #if tup in cache:
        #return cache[tup]

    hit_chance, max_hit = get_atk_stats(player, enemy)

    # Calculate how long it should take to kill one enemy
    ehtk = expected_htk(hit_chance, max_hit, enemy.hitpoints)
    expected_ttk = ehtk * player.equip.weapon.weapon.attack_speed * 0.6
    print('computed htk: %.1f' % ehtk)

    # add a delay for finding, targeting a new enemy
    expected_ttk += KILL_DELAY

    #cache[tup] = expected_ttk
    return expected_ttk


def expected_dmg(player, enemy, time):
    # now do the enemy
    hit_chance, max_hit = get_def_stats(player, enemy)
    dps = hit_chance * (max_hit + 1) / 2 / (enemy.attack_speed * 0.6)
    dmg = dps * time

    return dmg


def simulate(player, enemy):
    hit_chance, max_hit = get_atk_stats(player, enemy)
    htk_weights = simulate_htk(hit_chance, max_hit, enemy.hitpoints)

    # now do the enemy
    hit_chance, max_hit = get_def_stats(player, enemy)

    # convert player hits to enemy hits
    tick_weights = []
    p_speed = player.equip.weapon.weapon.attack_speed
    for i, w in enumerate(htk_weights):
        tick_weights.extend([(i, w)] * p_speed)

    hit_weights = []
    last_i = -1
    for i in range(0, len(tick_weights), enemy.attack_speed):
        if tick_weights[i][0] == last_i:
            hit_weights[-1] = tick_weights[i][1]
        else:
            hit_weights.append(tick_weights[i][1])
        last_i = tick_weights[i][0]

    sdmg = simulate_damage(hit_chance, max_hit, hit_weights)
    print('simulated dmg: %.1f' % sdmg)


"""
Find the expected xp per second of the given encounter.
"""
def xp_rate(player, monster):
    ettk = expected_ttk(player, monster)

    # TODO below here
    monster_dps = calc_dps()

    # add a delay for banking for new food
    if player.heal_rate < monster_dps:
        monsters_per_bank = player_hp / expected_ttk * (monster_dps - player_heal_rate)
        ettk += BANK_DELAY / monsters_per_bank

    ev = monster.hitpoints * 4. / ettk


"""
For a given attack level, strength level, and training style, find the fastest
possible way to get to the next level
"""
def best_rate(attack, strength, aggressive, monsters=all_monsters):
    best_rate = 0
    best_combo = None

    player = Player(attack=attack, strength=strength)

    for weapon in all_weapons:
        # make sure the player has the required levels to weild the weapon
        if (weapon.equipment.requirements and
            (player.attack < weapon.equipment.requirements.get('attack', 1) or
             player.strength < weapon.equipment.requirements.get('strength', 1))):
            continue

        for stance in weapon.weapon.stances:
            if (aggressive and stance['experience'] == 'strength' or
                (not aggressive) and stance['experience'] == 'attack'):
                for monster in monsters:
                    rate = xp_rate(player, monster, weapon, stance)
                    if rate > best_rate:
                        best_rate = rate
                        best_combo = (weapon.name, stance['combat_style'],
                                      '%s lvl %d' % (monster.name,
                                                     monster.combat_level))

    return best_combo, best_rate


"""
How many XP are required from level lvl - 1 to lvl?
"""
def xp_req(lvl):
    return int(lvl - 1 + 300 * 2 ** ((lvl - 1) / 7.)) / 4.


"""
How many XP are required from level lv1 to lv2?
"""
def xp_diff(lv1, lv2):
    total = 0
    for lvl in range(lv1 + 1, lv2 + 1):
        total += xp_req(lvl)
    return total


"""
Given starting (attack, strength) and target (attack, strength), find the
fastest way to train.
"""
def best_path(start_atk, start_str, end_atk, end_str, monsters=all_monsters):
    best_paths = {(start_atk, start_str): (0, None, None, None)}
    q = [(start_atk, start_str)]
    seen = set()
    while q:
        a, s = q.pop(0)

        if a == end_atk and s == end_str:
            break

        if a < end_atk:
            atk_combo, atk_rate = best_rate(a, s, False, monsters)
            atk_cost = best_paths[(a, s)][0] + xp_req(a + 1) / atk_rate
            if (a + 1, s) not in best_paths or atk_cost < best_paths[(a + 1, s)][0]:
                best_paths[(a + 1, s)] = (atk_cost, atk_combo,
                                          (a, s), (a + 1, s))

            if (a+1, s) not in seen:
                q.append((a+1, s))
                seen.add((a+1, s))

        if s < end_str:
            str_combo, str_rate = best_rate(a, s, True, monsters)
            str_cost = best_paths[(a, s)][0] + xp_req(s + 1) / str_rate
            if (a, s + 1) not in best_paths or str_cost < best_paths[(a, s + 1)][0]:
                best_paths[(a, s + 1)] = (str_cost, str_combo,
                                          (a, s), (a, s + 1))

            if (a, s + 1) not in seen:
                q.append((a, s+1))
                seen.add((a, s+1))

    path = [best_paths[(end_atk, end_str)]]
    while True:
        tup = path[-1][2]
        if tup is None:
            break
        path.append(best_paths[tup])

    steps = path[::-1]
    phases = []
    for i in range(1, len(steps)):
        td = timedelta(seconds=steps[i][0] - steps[i-1][0])
        if steps[i][1] != steps[i-1][1]:
            phases.append([steps[i][1], steps[i][2], steps[i][3], td])
        else:
            phases[-1][2] = steps[i][3]
            phases[-1][3] += td

    for p in phases:
        start, end = p[1], p[2]
        if end[0] > start[0]:
            message = 'attack from %d to %d' % (start[0], end[0])
        else:
            message = 'strength from %d to %d' % (start[1], end[1])

        print(p[0], message, p[3])

    print('total time:', timedelta(seconds=steps[-1][0]))
