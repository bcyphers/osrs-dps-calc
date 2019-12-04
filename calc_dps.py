import random
from datetime import timedelta
#from osrsbox import items_api, monsters_api

#all_db_monsters = monsters_api.load()

SLASH = 'slash'
STAB = 'stab'
CRUSH = 'crush'

class Enemy(object):
    def __init__(self, name, hp, def_lvl, stab, slash, crush):
        self.name = name
        self.hp = hp
        self.def_lvl = def_lvl
        self.stab = stab
        self.slash = slash
        self.crush = crush

    def max_def_roll(self, dmg_type):
        equip_bonus = getattr(self, dmg_type)
        return (self.def_lvl + 9) * (equip_bonus + 64)

chicken = Enemy('Chicken', 3, 1, -42, -42, -42)
cow = Enemy('Cow', 8, 1, -21, -21, -21)
goblin = Enemy('Goblin', 5, 1, -15, -15, -15)
hill_giant = Enemy('Hill Giant', 35, 26, 0, 0, 0)

enemies = [
    chicken,
    cow,
    goblin,
    hill_giant
]


class Weapon(object):
    def __init__(self, name, level, strength, stab, slash, crush, speed,
                 controlled, aggressive):
        self.name = name
        self.level = level
        self.strength = strength
        self.stab = stab
        self.slash = slash
        self.crush = crush
        self.interval = speed * 0.6
        self.controlled = controlled
        self.aggressive = aggressive

    def attack(self, aggressive=False):
        if aggressive:
            return getattr(self, self.aggressive)
        else:
            return getattr(self, self.controlled)

iron_scim = Weapon('Iron Scimitar', 1, 9, 2, 10, -2, 4, SLASH, SLASH)
iron_2h = Weapon('Iron 2h', 1, 14, -4, 13, 10, 7, SLASH, SLASH)

black_scim = Weapon('Black Scimitar', 10, 14, 4, 19, -2, 4, SLASH, SLASH)
black_2h = Weapon('Black 2h', 10, 26, -4, 27, 21, 7, SLASH, SLASH)

mith_scim = Weapon('Mithril Scimitar', 20, 20, 5, 21, -2, 4, SLASH, SLASH)
mith_2h = Weapon('Mithril 2h', 20, 31, -4, 30, 24, 7, SLASH, SLASH)

addy_scim = Weapon('Addy Scimitar', 30, 28, 6, 29, -2, 4, SLASH, SLASH)
addy_2h = Weapon('Addy 2h', 30, 44, -4, 43, 30, 7, SLASH, SLASH)

rune_scim = Weapon('Rune Scimitar', 40, 44, 7, 45, -2, 6, SLASH, SLASH)
rune_2h = Weapon('Rune 2h', 40, 70, -4, 69, 50, 7, SLASH, SLASH)

weapons = [
    iron_scim, iron_2h,
    black_scim, black_2h,
    mith_scim, mith_2h,
    addy_scim, addy_2h,
    rune_scim, rune_2h
]


def expected_htk(hit_chance, max_hit, hp):
    dic = {}
    def T(n):
        if n in dic:
            return dic[n]

        if n <= 0:
            res = 0

        else:
            prevs = 0
            for i in range(n - max_hit, n):
                prevs += T(i)

            res = (1 / hit_chance) + (1. / max_hit) * prevs

        dic[n] = res
        return res

    return T(hp)


class Player(object):
    def __init__(self, attack=1, strength=1, weapon=iron_2h):
        self.attack = attack
        self.strength = strength
        self.weapon = weapon

    def expected_ttk(self, enemy, aggressive=False):
        effective_str = self.strength + aggressive * 3 + 8
        max_hit = int(0.5 + effective_str * (self.weapon.strength + 64) / 640)

        effective_atk = self.attack + (not aggressive) * 3 + 8
        max_atk_roll = effective_atk * (self.weapon.attack(aggressive) + 64)

        if aggressive:
            dmg_type = self.weapon.aggressive
        else:
            dmg_type = self.weapon.controlled

        enemy_def_roll = enemy.max_def_roll(dmg_type)
        if max_atk_roll > enemy_def_roll:
            hit_chance = 1 - enemy_def_roll / (2. * max_atk_roll)
        else:
            hit_chance = max_atk_roll / (3. * enemy_def_roll)

        return expected_htk(hit_chance, max_hit, enemy.hp) * self.weapon.interval + 3

    def xp_rate(self, enemy, aggressive=False):
        return enemy.hp * 4. / self.expected_ttk(enemy, aggressive)


def best_rate(attack, strength, aggressive):
    best_rate = 0
    best_combo = None

    for w in weapons:
        if attack < w.level:
            continue
        for e in enemies:
            rate = Player(attack, strength, w).xp_rate(e, aggressive)
            if rate > best_rate:
                best_rate = rate
                best_combo = w.name, e.name

    return best_combo, best_rate


def xp_req(lvl):
    return int(lvl - 1 + 300 * 2 ** ((lvl - 1) / 7.)) / 4.


def xp_diff(lv1, lv2):
    total = 0
    for lvl in range(lv1 + 1, lv2 + 1):
        total += xp_req(lvl)
    return total


def best_path(start_atk, start_str, end_atk, end_str):
    best_paths = {(start_atk, start_str): (0, None, None)}
    q = [(start_atk, start_str)]
    seen = set()
    while q:
        a, s = q.pop(0)

        if a == end_atk and s == end_str:
            break

        if a < end_atk:
            atk_combo, atk_rate = best_rate(a, s, False)
            atk_cost = best_paths[(a, s)][0] + xp_req(a + 1) / atk_rate
            if (a + 1, s) not in best_paths or atk_cost < best_paths[(a + 1, s)][0]:
                best_paths[(a + 1, s)] = (atk_cost, atk_combo, (a, s))


            if (a+1, s) not in seen:
                q.append((a+1, s))
                seen.add((a+1, s))

        if s < end_str:
            str_combo, str_rate = best_rate(a, s, True)
            str_cost = best_paths[(a, s)][0] + xp_req(s + 1) / str_rate
            if (a, s + 1) not in best_paths or str_cost < best_paths[(a, s + 1)][0]:
                best_paths[(a, s + 1)] = (str_cost, str_combo, (a, s))

            if (a, s + 1) not in seen:
                q.append((a, s+1))
                seen.add((a, s+1))

    path = [best_paths[(end_atk, end_str)]]
    while True:
        tup = path[-1][-1]
        if tup is None:
            break
        path.append(best_paths[tup])

    steps = path[::-1]
    for i in range(1, len(steps)):
        print(steps[i][2], steps[i][1],
            timedelta(seconds=steps[i][0] - steps[i-1][0]))

    print('total time:', timedelta(seconds=steps[-1][0]))

