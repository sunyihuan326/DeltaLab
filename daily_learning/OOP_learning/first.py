# coding:utf-8 
'''
created on 2018/5/3

@author:sunyihuan
'''


class hero:
    def __init__(self, health, mana):
        self._health = health
        self._mana = mana

    def attack(self) -> int:
        return 1

    def take_damage(self, damage: int):
        self._health -= damage
        return self._health

    def is_alive(self):
        return self._health > 0


h = hero(1, 2)
print(h.take_damage(2.2))
