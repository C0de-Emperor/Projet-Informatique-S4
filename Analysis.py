from typing import TYPE_CHECKING
from math import log10, sqrt
from Mathematics import DiscreteFunction

if TYPE_CHECKING:
    from Mathematics import ComplexDiscreteFunction


# Local
def MSE(x: "DiscreteFunction", y: "DiscreteFunction"):
    result = 0
    for i in range(x.width):
        for j in range(x.height):
            result += (x[i, j] - y[i, j])**2

    return result / (x.width * x.height)

def PSNR(x: "DiscreteFunction", y : "DiscreteFunction"):
    mse = MSE(x, y)

    if mse == 0:
        return float("inf")

    return 10 * log10((255**2) / mse)

# Global
def SSIM(x: DiscreteFunction, y: DiscreteFunction):
    """
    Iterpretation for SSIM

    - [0, 0.75] = Dégradation visible
    - [0.75, 0.95] = Correct
    - [0.95, 1] = Très bon
    - 1 = Identique
    """


    N = x.width * x.height

    # Moyennes
    mu_x = x.getExpectation()
    mu_y = y.getExpectation()

    # Variances et covariance
    var_x = 0
    var_y = 0
    cov_xy = 0

    for i in range(x.width):
        for j in range(x.height):
            dx = x[i, j] - mu_x
            dy = y[i, j] - mu_y

            var_x += dx**2
            var_y += dy**2
            cov_xy += dx*dy

    var_x /= N
    var_y /= N
    cov_xy /= N

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    numerator = (2*mu_x*mu_y + C1) * (2*cov_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2)

    return numerator / denominator

def LaplacianVariance (x: DiscreteFunction):
    """
    Interpretation
    
    Un résultat élevé indique de grandes variations dans l'image. 
    Plus une image possède des contours, plus cette variation est grande. 
    Elle aura donc moins de chances d'être considérée comme floue.

    si sup 215 => nette
    """

    LAPLACIAN_KERNEL = DiscreteFunction([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])

    result = x.convolve(LAPLACIAN_KERNEL)

    return result.Variance()

def SobelVariance (x: DiscreteFunction):
    """
    Interpretation
    
    Un résultat élevé indique de grandes variations dans l'image. 
    Plus une image possède des contours, plus cette variation est grande. 
    Elle aura donc moins de chances d'être considérée comme floue.

    si sup a (1200, 2650, 80) => nette
    """

    X_KERNEL = DiscreteFunction([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    
    Y_KERNEL = DiscreteFunction([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    XY_KERNEL = DiscreteFunction([
        [1, 0, -1],
        [0, 0, 0],
        [-1, 0, 1]
    ])

    Var_x = x.convolve(X_KERNEL).Variance()
    Var_y = x.convolve(Y_KERNEL).Variance()
    Var_xy = x.convolve(XY_KERNEL).Variance()

    return (Var_x, Var_y, Var_xy)

def RMS (x: DiscreteFunction):
    """
    Interpretation

    Donne une première évaluation du contraste de l'image

    bien si [45, 75]
    """

    return x.StandardDeviation()




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

def ChooseFilter(img):
    var = LocalVariance(img)
    grad = GradientEnergy(img)
    #hf = HighFrequencyRatio(img)

    if var > 500 and grad < 50:
        return "gaussian"

    if var > 1000: #and hf > 0.6:
        return "median"

    if grad > 200 and var > 800:
        return "bilateral"

    return "adaptive_gaussian"







