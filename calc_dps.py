import random
from datetime import timedelta

from osrsbox import items_api, monsters_api
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
    return (not i.quest_item and i.equipable_by_player and not i.placeholder
            and i.equipment and i.highalch > 0)


api_items = items_api.load()
all_equipment = [i for i in api_items if valid_equipment(i)]
p2p_weapons = [i for i in all_equipment if i.weapon]
f2p_weapons = [i for i in p2p_weapons if (not i.members) and i.tradeable]

weapon_dict = {}
for w in p2p_weapons:
    weapon_dict[w.name.lower()] = w

all_weapons = list(weapon_dict.values())

SLOTS = ['head', 'body', 'legs', 'feet', 'hands', 'cape', 'neck', 'ring',
         'shield']

STATS = ['attack_stab', 'attack_slash', 'attack_crush', 'attack_magic',
         'attack_ranged', 'defence_stab', 'defence_slash', 'defence_crush',
         'defence_magic', 'defence_ranged', 'melee_strength',
         'ranged_strength', 'magic_damage', 'prayer']

all_armor = {s: [] for s in SLOTS}

for i in all_equipment:
    if i.equipment.slot in SLOTS:
        all_armor[i.equipment.slot].append(i)

cache = {}


class AttackStyle(object):
    SLASH = 'slash'
    STAB = 'stab'
    CRUSH = 'crush'
    MAGIC = 'magic'
    RANGE = 'range'


class ArmorSet(object):
    def __init__(self, **kwargs):
        for s in SLOTS:
            setattr(self, s, kwargs.get(s))

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
    def __init__(self, attack, strength, defence, ranged, magic, prayer):
        self.attack = attack
        self.strength = strength
        self.defence = defence
        self.ranged = ranged
        self.magic = magic
        self.prayer = prayer


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

    return ArmorSet(**aset)


class Encounter(object):
    def __init__(self, player, monster):
        self.player = player
        self.monster = monster


def get_atk_stats(attack, strength, enemy, weapon, stance):
    # Calculate strength and attack bonuses due to combat stance
    str_bonus = 0
    atk_bonus = 0
    if stance['attack_style'] == 'accurate':
        atk_bonus = 3
    elif stance['attack_style'] == 'controlled':
        str_bonus = 1
        atk_bonus = 1
    elif stance['attack_style'] == 'aggressive':
        str_bonus = 3

    # Calculate the player's max hit
    effective_str = strength + str_bonus + 8
    max_hit = int(0.5 + effective_str * (weapon.equipment.melee_strength + 64) / 640)

    # Calculate the player's max attack roll
    effective_atk = attack + atk_bonus + 8
    weapon_atk = getattr(weapon.equipment, "attack_" + stance['attack_type'])
    max_atk_roll = effective_atk * (weapon_atk + 64)

    # Calculate the enemy's max defence roll
    enemy_equip_bonus = getattr(enemy, "defence_" + stance['attack_type'])
    enemy_def_roll =  (enemy.defence_level + 9) * (enemy_equip_bonus + 64)

    # Calculate the chance of a hit (not a splash)
    if max_atk_roll > enemy_def_roll:
        hit_chance = 1 - enemy_def_roll / (2. * max_atk_roll)
    else:
        hit_chance = max_atk_roll / (3. * enemy_def_roll)

    print('max hit: %d, hit chance: %.3f' % (max_hit, hit_chance))

    return hit_chance, max_hit

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


def simulate_htk(hit_chance, max_hit, hp, n=10000):
    res = []
    for i in range(n):
        health = hp
        hits = 0
        while health > 0:
            hits += 1
            if random.random() < hit_chance:
                health -= random.randint(1, max_hit)

        res.append(hits)

    plt.hist(res)
    plt.show()

    mean = sum(res) / float(n)
    ranked = sorted(res)
    med = ranked[n//2]
    low_bound = ranked[n//20]
    hi_bound = ranked[-n//20]

    print('%d < %d < %d' % (low_bound, med, hi_bound))
    return mean


KILL_DELAY = 3
BANK_DELAY = 150

"""
Compute expected time to kill a enemy given attack level, strength level,
weapon, and weapon stance.
"""
def expected_ttk(attack, strength, enemy, weapon, stance):
    tup = (attack, strength, enemy.id, weapon.id, stance['combat_style'])
    if tup in cache:
        return cache[tup]

    hit_chance, max_hit = get_atk_stats(attack, strength, enemy, weapon, stance)

    # Calculate how long it should take to kill one enemy
    ehtk = expected_htk(hit_chance, max_hit, enemy.hitpoints)
    expected_ttk = ehtk * weapon.weapon.attack_speed * 0.6

    shtk = simulate_htk(hit_chance, max_hit, enemy.hitpoints)
    print('computed: %.1f, simulated: %.1f' % (ehtk, shtk))

    # add a delay for finding, targeting a new enemy
    expected_ttk += KILL_DELAY

    cache[tup] = expected_ttk
    return expected_ttk


"""
Find the expected xp per second of the given encounter.
"""
def xp_rate(player, monster, weapon, stance):
    ettk = expected_ttk(player.attack, player.strength, monster, weapon, stance)

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
