from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Mathematics import DiscreteFunction, ComplexDiscreteFunction

# --- Indicateurs de dégradation --- #

def LocalVariance(function: "DiscreteFunction", window: int = 3):
    """
    Interprétation :
    - ↑ variance = ↑ bruit
    - ↓ variance = image plus propre (ou trop floue)
    """

    w = window // 2
    H, W = function.height, function.width
    total = 0
    count = 0

    for i in range(W):
        for j in range(H):
            vals = [
                function[i+m, j+n]
                for m in range(-w, w+1)
                for n in range(-w, w+1)
            ]
            mean = sum(vals) / len(vals)
            total += sum((v - mean)**2 for v in vals) / len(vals)
            count += 1

    return total / count

def GradientEnergy(function: "DiscreteFunction"):
    """
    Interprétation :
    - ↑ énergie → contours nets ou bruitage très fort
    - ↓ énergie → flou
    """

    total = 0
    for i in range(function.width):
        for j in range(function.height):
            gx = function[i+1, j] - function[i-1, j]
            gy = function[i, j+1] - function[i, j-1]
            total += gx*gx + gy*gy
    return total / (function.width * function.height)

def HighFrequencyRatio(function: "ComplexDiscreteFunction"):
    """
    Interprétation :
    - bruit = beaucoup de HF
    - flou = peu de HF
    """

    r0 = min(function.height, function.width) // 6

    total = 0
    high = 0

    for u in range(function.width):
        for v in range(function.height):
            mag = abs(function[u, v])
            total += mag
            if abs(u-function.width//2) + abs(v-function.height//2) > r0:
                high += mag

    return high / total

# --- Indicateurs d’amélioration --- #

def NoiseReduction(before, after):
    return LocalVariance(before) - LocalVariance(after)

def EdgePreservation(before, after):
    """
    Interprétation :
    - = 1 → contours préservés
    - < 0.7 → trop flou
    """
    return GradientEnergy(after) / GradientEnergy(before)

# --- Indicateurs d’etat --- #

def QualityScore(function : "DiscreteFunction"):
    from math import exp
    return exp(GradientEnergy(function) / (LocalVariance(function) + 1e-6))/2





















