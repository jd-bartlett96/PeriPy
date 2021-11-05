"""Damage laws.

TODO: also define damage laws for composite materials, here.
"""
import numpy as np


def bond_damage_PMB(
        stretch, sc, bond_damage):
    """
    Calculate the bond softening factors for the trilinear model.

    Also known as ``bond damge'', the bond softening factors are applied to
    satisfy the damage law.

    :arg int global_size: The number of bonds.
    :arg stretch:
    :type stretch:
    :arg float s0:
    :arg float s1:
    :arg float sc:
    :arg bond_damage:
    :type bond_damage:
    :arg float beta:
    """
    # bond softening factors will not increase from 0 under linear elastic
    # regime
    if stretch < sc:
        bond_damage_temp = 0.0
    else:
        bond_damage_temp = 1.0
    # bond softening factor can only increase (damage is irreversible)
    if bond_damage_temp > bond_damage:
        bond_damage = bond_damage_temp
    return bond_damage


def bond_damage_sigmoid(
        stretch, sc, sigma, bond_damage):
    """
    Calculate the bond softening factors for the sigmoid model.

    Also known as ``bond damge'', the bond softening factors are applied to
    satisfy the damage law.
    """
    bond_damage = 1 / (
        np.exp((stretch - sc) / sigma) + 1)
    return bond_damage


def bond_damage_trilinear(
        stretch, s0, s1, sc, bond_damage, beta):
    """
    Calculate the bond softening factors for the trilinear model.

    Also known as ``bond damge'', the bond softening factors are applied to
    satisfy the damage law.

    :arg int global_size: The number of bonds.
    :arg stretch:
    :type stretch:
    :arg float s0:
    :arg float s1:
    :arg float sc:
    :arg bond_damage:
    :type bond_damage:
    :arg float beta:
    """
    eta = s1 / s0
    # bond softening factors will not increase from 0 under linear elastic
    # loading, stretch[bond] <= s0
    if (stretch > s0) and (stretch <= s1):
        bond_damage_temp = (
            1 - ((eta - beta) / (eta - 1) * (s0 / stretch))
            + ((1 - beta) / (eta - 1)))
    elif (stretch > s1) and (stretch <= sc):
        bond_damage_temp = 1 - (
                (s0 * beta / stretch)
                * ((sc - stretch) / (sc - s1)))
    elif stretch > sc:
        bond_damage_temp = 1
    # Bond softening factor can only increase (damage is irreversible)
    if bond_damage_temp > bond_damage:
        bond_damage = bond_damage_temp
    return bond_damage


def bond_damage_exponential(
        stretch, s0, sc, bond_damage, k, alpha):
    """
    Calculate the bond softening factors for the trilinear model.

    Also known as ``bond damge'', the bond softening factors are applied to
    satisfy the damage law.

    :arg int global_size: The number of bonds.
    :arg stretch:
    :type stretch:
    :arg float s0:
    :arg float sc:
    :arg bond_damage:
    :type bond_damage:
    :arg float k:
    :arg float alpha:
    """
    if (stretch > s0) and (stretch < sc):
        numerator = 1 - np.exp(-k * (stretch - s0) / (sc - s0))
        residual = alpha * (1 - (stretch - s0) / (sc - s0))
        bond_damage_temp = 1 - (s0 / stretch) * (
            (1 - numerator / (1 - np.exp(-k))) + residual) / (1 + alpha)
    elif stretch > sc:
        bond_damage_temp = 1
    # Bond softening factor can only increase (damage is irreversible)
    if bond_damage_temp > bond_damage:
        bond_damage = bond_damage_temp
    return bond_damage
