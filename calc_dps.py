import random
from datetime import timedelta
from osrsbox import items_api, monsters_api

def monster_complete(m):
    return m.defence_level >= 1 and m.hitpoints >= 1

api_monsters = monsters_api.load()
p2p_monsters = [m for m in api_monsters if monster_complete(m)]
f2p_monsters = [m for m in api_monsters if monster_complete(m) and not m.members]
monster_dict = {}
for m in f2p_monsters:
    if (m.name.lower(), m.combat_level) not in monster_dict:
        monster_dict[(m.name.lower(), m.combat_level)] = m
all_monsters = list(monster_dict.values())

api_items = items_api.load()
f2p_weapons = [i for i in api_items if i.weapon and (not i.members)
               and i.tradeable]
weapon_dict = {}
for w in f2p_weapons:
    if any(s in w.name.lower() for s in ['rune', 'adamant', 'mithril',
                                         'black', 'steel', 'iron']):
        if any(s in w.name.lower() for s in ['dagger', 'sword', 'battleaxe',
                                             'scimitar', 'warhammer']):
            weapon_dict[w.name.lower()] = w
all_weapons = list(weapon_dict.values())

cache = {}

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

KILL_DELAY = 3
BANK_DELAY = 150

"""
Compute expected time to kill a monster given attack level, strength level,
weapon, and weapon stance.
"""
def expected_ttk(attack, strength, monster, weapon, stance):
    tup = (attack, strength, monster.id, weapon.id, stance['combat_style'])
    if tup in cache:
        return cache[tup]

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

    # Calculate the monster's max defence roll
    monster_equip_bonus = getattr(monster, "defence_" + stance['attack_type'])
    monster_def_roll =  (monster.defence_level + 9) * (monster_equip_bonus + 64)

    # Calculate the chance of a hit (not a splash)
    if max_atk_roll > monster_def_roll:
        hit_chance = 1 - monster_def_roll / (2. * max_atk_roll)
    else:
        hit_chance = max_atk_roll / (3. * monster_def_roll)

    # Calculate how long it should take to kill one monster
    expected_ttk = expected_htk(hit_chance, max_hit, monster.hitpoints) * \
        weapon.weapon.attack_speed * 0.6

    # add a delay for finding, targeting a new monster
    expected_ttk += KILL_DELAY

    cache[tup] = expected_ttk
    return expected_ttk


"""
Find the expected xp per second of the given encounter.
"""
def xp_rate(player, monster, weapon, stance):
    ettk = expected_ttk(player.attack, player.strength, monster, weapon, stance)

    monster_dps = calc_dps(player.

    # add a delay for banking for new food
    if player.heal_rate < monster_dps:
        monsters_per_bank = player_hp / expected_ttk * (monster_dps - player_heal_rate)
        expected_ttk += BANK_DELAY / monsters_per_bank

    ev = monster.hitpoints * 4. / ettk


"""
For a given attack level, strength level, and training style, find the fastest
possible way to get to the next level
"""
def best_rate(attack, strength, aggressive, monsters=all_monsters):
    best_rate = 0
    best_combo = None

    for weapon in all_weapons:
        # make sure the player has the required levels to weild the weapon
        if (weapon.equipment.requirements and
            (attack < weapon.equipment.requirements.get('attack', 1) or
             strength < weapon.equipment.requirements.get('strength', 1))):
            continue

        for stance in weapon.weapon.stances:
            if (aggressive and stance['experience'] == 'strength' or
                (not aggressive) and stance['experience'] == 'attack'):
                for monster in monsters:
                    rate = xp_rate(attack, strength, monster, weapon, stance)
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
