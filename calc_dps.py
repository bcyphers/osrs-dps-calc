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
            and not i.placeholder and i.equipment and i.highalch is not None)

SLOTS = ['weapon', 'head', 'body', 'legs', 'feet', 'hands', 'cape', 'neck',
         'ring', 'shield']

STATS = ['attack_stab', 'attack_slash', 'attack_crush', 'attack_magic',
         'attack_ranged', 'defence_stab', 'defence_slash', 'defence_crush',
         'defence_magic', 'defence_ranged', 'melee_strength',
         'ranged_strength', 'magic_damage', 'prayer']

PLAYER_SKILLS = [
    'attack',
    'strength',
    'defence',
    'ranged',
    'magic',
    'prayer',
    'hitpoints',
]


api_items = items_api.load()
all_equipment = [i for i in api_items if valid_equipment(i)]
p2p_weapons = [i for i in all_equipment if i.weapon]
f2p_weapons = [i for i in p2p_weapons if (not i.members) and i.tradeable]

weapon_dict = {}
for w in p2p_weapons:
    weapon_dict[w.name.lower()] = w

all_weapons = list(weapon_dict.values())

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
                 prayer=1, hitpoints=10, equipment=None):
        self.attack = attack
        self.strength = strength
        self.defence = defence
        self.ranged = ranged
        self.magic = magic
        self.prayer = prayer
        self.hitpoints = hitpoints

        if equipment is None:
            equipment = Equipment(0)

        self.equip = equipment

    def can_equip(self, gear):
        reqs = gear.equipment.requirements
        if reqs:
            for level in PLAYER_SKILLS:
                if getattr(self, level) < reqs.get(level, 1):
                    return False

        return True

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


###########################################################
## Basic encounter calculators
###########################################################

"""
Find out the player's max hit and hit chance versus a specific enemy.
"""
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

    return hit_chance, max_hit


"""
Find out an enemy's max hit and hit chance versus a player.
"""
def get_def_stats(player, enemy):
    hit_chance = None
    max_hit = None

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

    return hit_chance, max_hit


###########################################################
## Deep probability estimators
###########################################################

def convolve(a, b, cap=None):
    # the result will either be the sum of the vector lengths or the cap,
    # whichever is less
    if cap is not None:
        res_len = min(cap, len(a) + len(b) - 1)
    else:
        res_len = len(a) + len(b) - 1

    results = np.zeros(res_len)
    for i in range(len(a)):
        for j in range(len(b)):
            # the cap is going to be the max hp of a monster. so all
            # values that fall above that cap can be squished into the last box
            rix = min(i + j, res_len - 1)
            results[rix] += a[i] * b[j]

    return results


def simulate_htk(hit_chance, max_hit, hp):
    # this is the per-hit damage probability dist
    probs = [1 - hit_chance] + [hit_chance / max_hit] * max_hit

    # there is a 100% probability that 0 damage will have been done before the
    # first hit
    last = [1]

    # this array has the probability distribution for the total amount of damage
    # done after each hit. n_hit_dist[i][j] is the probability that exactly j
    # damage has been done after the ith hit
    n_hit_dist = [last]

    # res[i] is the probability that the ith hit ends the fight
    res = [0]

    # do this until we are 99.99% sure we have killed the enemy (it's always
    # possible that you splat forever)
    while sum(res) < 0.9999:
        # convolve the last set of probabilities that we didn't get lethal with
        # the damage probabilities for the next hit
        last = convolve(last[:hp], probs, cap=(hp + 1))
        n_hit_dist.append(last)

        # if the newest dist has a value in the `hp`th index, then there's a
        # chance that the fight is over. add it to the result
        if len(last) <= hp:
            res.append(0)
        else:
            res.append(last[hp])

    # let's see it
    plt.bar(range(len(res)), res)
    plt.show()

    return res


"""
hit_chance: probability that a hit will not splash
max_hit: max hit
hit_probs: for each index N, probability that this fight will last exactly N hits
"""
def simulate_damage(hit_chance, max_hit, hit_probs):
    # index = damage, value = probability of that much damage in one hit
    probs = [1 - hit_chance] + [hit_chance / max_hit] * max_hit

    # index = number of hits, value = distribution of total damage done
    n_hit_dist = [probs]

    # create the n_hit_dist matrix by convolving the hit probability vector over
    # and over
    for i in range(len(hit_probs) - 1):
        n_hit_dist.append(convolve(n_hit_dist[-1], probs))

    # result vector will be as long as the longest hit distribution (the last
    # one)
    res = np.zeros(len(n_hit_dist[-1]))

    # weight each hit dist vector by how likely it is to be the last one
    for i, p in enumerate(hit_probs):
        padded_hit_dist = np.append(n_hit_dist[i],
                                    np.zeros(len(res) - len(n_hit_dist[i])))
        res += p * padded_hit_dist

    # the mean damage value should just be the sum of the weighted values
    mean = 0
    for i in range(len(res)):
        mean += i * res[i]

    cdf = [sum(res[:i+1]) for i in range(len(res))]
    med = next(i for i, x in enumerate(cdf) if x > 0.5)
    low_bound = next(i for i, x in enumerate(cdf) if x > 0.05)
    hi_bound = next(i for i, x in enumerate(cdf) if x > 0.95)
    hi_hi_bound = next(i for i, x in enumerate(cdf) if x > 0.9999)

    print('mean simulated damage: %.1f' % mean)
    print('medians: %d (5%%) < %d (50%%) < %d (95%%)' %
          (low_bound, med, hi_bound))

    # cut off the very unlikely high values and lump them together
    out = res[:hi_hi_bound]
    np.append(out, sum(res[hi_hi_bound:]))
    plt.bar(range(len(out)), out)
    plt.show()

    return out


"""
Figure out the exact(ish) distribution of (1) hits to kill the enemy and (2)
damage done by the enemy
"""
def simulate(player, enemy, aggro=True):
    # get player's hit chance and max hit
    hit_chance, max_hit = get_atk_stats(player, enemy)

    print('player max hit: %d, hit chance: %.1f%%' % (max_hit, hit_chance * 100))

    # calculate the distribution of the number of hits the player will take to
    # kill the enemy (in the form of a discrete probability distribution function)
    htk_pdf = simulate_htk(hit_chance, max_hit, enemy.hitpoints)

    # now do the enemy's max hit and hit chance
    hit_chance, max_hit = get_def_stats(player, enemy)

    print('enemy max hit: %d, hit chance: %.1f%%' % (max_hit, hit_chance * 100))

    # Now we need to convert player hits to enemy hits. In other words, in the
    # time it takes the player to get off X hits, the enemy can get off Y hits.

    # first, convert the hits-to-kill PDF to a ticks-to-kill cumulative
    # distribution function (CDF). This will have a lot of duplicates.
    tick_cdf = []
    p_speed = player.equip.weapon.weapon.attack_speed
    htk_cdf = [sum(htk_pdf[:i]) for i in range(len(htk_pdf))]
    for w in htk_cdf:
        tick_cdf.extend([w] * p_speed)

    # next convert the per-tick CDF to a per-monster-hit CDF. The numbers in
    # hit_cdf are the probability that the fight has ended before the Nth hit by
    # the monster.

    # if the enemy started the fight, they have a head start
    if aggro:
        start = 0
    else:
        start = p_speed + p

    # pull out the cdf values for the relevant ticks
    hit_cdf = tick_cdf[start::enemy.attack_speed]

    # finally, convert this CDF to a PDF which represents the probability that a
    # monster will get exactly N hits on the player during the fight (the index
    # is N).
    hit_pdf = []
    for i, cur_p in enumerate(hit_cdf):
        if i + 1 < len(hit_cdf):
            # this is the probability that the fight will have ended after the
            # next tick
            next_p = hit_cdf[i+1]
        else:
            next_p = 1

        # we're looking for the difference between the cdf on this hit and
        # the cdf on the next hit
        hit_pdf.append(next_p - cur_p)

    # now pass this distribution into the damage simulator to figure out how
    # much the enemy is likely to damage the player during their fight.
    dmg_pdf = simulate_damage(hit_chance, max_hit, hit_pdf)

    return htk_pdf, dmg_pdf


###########################################################
## Expected value estimators (fast)
###########################################################

KILL_DELAY = 3
BANK_DELAY = 150
PLAYER_HEAL_RATE = 1. / 60

"""
Calculate the expected number of hits to kill an enemy, given hit chance, max
hit, and the enemy's HP.
"""
def expected_htk(hit_chance, max_hit, hp):
    # each element in the array will be the expected number of hits to do at
    # least that much damage. e.g. ehtk[0] => the number of hits expected to do
    # at least 1 damage.
    ehtk = []

    for i in range(hp):
        prevs = ehtk[max(i - max_hit, 0):]
        res = (1 / hit_chance) + (1. / max_hit) * sum(prevs)
        ehtk.append(res)

    # the last value in the array is the expected number of hits to kill
    return ehtk[-1]


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
    ettk = ehtk * player.equip.weapon.weapon.attack_speed * 0.6

    # add a delay for finding, targeting a new enemy
    ettk += KILL_DELAY

    #cache[tup] = ettk
    return ettk


def expected_dps(player, enemy):
    # what are the enemy's max hit and hit chance against the player?
    hit_chance, max_hit = get_def_stats(player, enemy)

    if hit_chance is None:
        return None

    # compute enemy DPS
    return hit_chance * (max_hit + 1) / 2 / (enemy.attack_speed * 0.6)


"""
Compute the expected amount of damage an enemy will do to a player before the
player can kill it
"""
def expected_dmg(player, enemy, time, aggro=True):
    dps = expected_dps(player, enemy)

    # TODO: if the enemy attacks the player, it gets a small time advantage
    dmg = dps * time

    return dmg


"""
Find the expected xp per second of the given encounter.
"""
def xp_rate(player, enemy):
    ettk = expected_ttk(player, enemy)

    enemy_dps = expected_dps(player, enemy)
    if enemy_dps is None:
        return 0

    # add a delay for banking for new food
    if enemy_dps > PLAYER_HEAL_RATE:
        total_hp = player.hitpoints + 27 * 10
        dmg_per_enemy = ettk * (enemy_dps - PLAYER_HEAL_RATE)
        enemies_per_bank = total_hp / dmg_per_enemy
        ettk += BANK_DELAY / enemies_per_bank

    return enemy.hitpoints * 4. / (ettk / 3600)


###########################################################
## Optimizers
###########################################################

"""
Find the best equipment available for a player with a given set of stats and
attack style (slash, crush, stab, range, or mage)

TODO
"""
def find_bis_armor(player, attack_style):
    aset = {s: None for s in SLOTS}
    for slot, dic in equipment_dict.items():
        if slot == 'weapon':
            continue

        eligible = []
        for i in dic.values():
            elig = True
            for stat in PLAYER_SKILLS:
                req = i.equipment.requirements.get(stat, 1)
                if getattr(player, stat) < req:
                    elig = False
                    break

            if elig:
                eligible.append(i)

        attr = 'defence_' + attack_style
        aset[slot] = max(items, key=lambda i: getattr(i.equipment, attr))

    return Equipment(**aset)


"""
For a given set of stats, training style, and target monster, find the weapon
that will gain XP fastest.

stat must be one of 'strength', 'attack', 'defence'.
"""
def best_weapon(player, monster, stat='attack', n_results=10):
    results = []

    for weapon in all_weapons:
        # make sure the player has the required levels to weild the weapon
        if not player.can_equip(weapon):
            continue

        for stance in weapon.weapon.stances:
            if stance['experience'] == stat or \
                    stance['experience'] == 'balanced':
                #player.equip = find_bis_armor(player, stance['attack_style'])
                player.equip.weapon = weapon
                player.equip.stance = stance

                rate = xp_rate(player, monster)
                combo = (weapon.name, stance['combat_style'])
                results.append((rate, combo))

    return list(sorted(results))[:-n_results:-1]


"""
For a given player, find the monster that will let them train XP fastest.
"""
def best_monster(player, monsters=all_monsters, n_results=10):
    results = []

    for monster in monsters:
        if not monster.attack_speed:
            continue

        rate = xp_rate(player, monster)
        monster = ('%s lv. %d' % (monster.name, monster.combat_level))
        results.append((rate, monster))

    return list(sorted(results))[:-n_results:-1]


"""
How many XP are required from level (lvl - 1) to lvl?
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
