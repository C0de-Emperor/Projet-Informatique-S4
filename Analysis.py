from typing import TYPE_CHECKING
from math import log10, sqrt
from Mathematics import DiscreteFunction
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from Mathematics import ComplexDiscreteFunction


# --- Local ---
def MSE(source: "DiscreteFunction", image: "DiscreteFunction"):
    result = 0
    for i in range(source.width):
        for j in range(source.height):
            result += (source[i, j] - image[i, j])**2

    return result / (source.width * source.height)

def PSNR(source: "DiscreteFunction", image: "DiscreteFunction"):
    """
    Basé sur le calcul direct de l'écart pixel par pixel de chaque image
    """

    mse = MSE(source, image)

    if mse == 0:
        return float("inf")

    return 10 * log10((255**2) / mse)

# --- Global ---

# Quality
def SSIM(x: DiscreteFunction, y: DiscreteFunction):
    """
    Iterpretation for SSIM :

    - [0, 0.75] = Dégradation visible
    - [0.75, 0.95] = Correct
    - [0.95, 1] = Très bon
    - 1 = Identique

    Description :
    - le SSIM se concentre plus sur des caractéristiques globales de l'image en considérant les valeurs statistiques de celle-ci
    """

    N = x.width * x.height

    # Moyennes
    mu_x = x.MeanIntensity()
    mu_y = y.MeanIntensity()

    # Variances et covariance
    var_x = 0
    var_y = 0
    cov_xy = 0

    for i in range(x.width):
        for j in range(x.height):
            dx = x[i, j] - mu_x
            dy = y[i, j] - mu_y

            var_x += dx * dx
            var_y += dy * dy
            cov_xy += dx * dy

    var_x /= N
    var_y /= N
    cov_xy /= N

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    numerator = (2*mu_x*mu_y + C1) * (2*cov_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2)

    return numerator / denominator


# Flou
def LaplacianVariance (x: DiscreteFunction):
    """
    Interpretation :
    
    > Un résultat élevé indique de grandes variations dans l'image. 
    Plus une image possède des contours, plus cette variation est grande. 
    Elle aura donc moins de chances d'être considérée comme floue.

    Reactions :
    - salt&pepper = très grande variation negative, "la nouvelle image est plus nette que l'ancienne"
    - random noising = grande variation negative, "la nouvelle image est plus nette que l'ancienne"
    - gaussian = baisse de la valeur
    """

    LAPLACIAN_KERNEL = DiscreteFunction([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])

    result = x.convolve(LAPLACIAN_KERNEL)

    return result.IntensityVariance()

def SobelVariance (x: DiscreteFunction):
    """
    Interpretation :
    
    > Un résultat élevé indique de grandes variations dans l'image. 
    Plus une image possède des contours, plus cette variation est grande. 
    Elle aura donc moins de chances d'être considérée comme floue.

    Reactions :
    - salt&pepper = variation negative, "la nouvelle image est plus nette que l'ancienne"
    - random noising = variation negative, "la nouvelle image est plus nette que l'ancienne"
    - gaussian = baisse de la valeur
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

    Var_x = x.convolve(X_KERNEL).IntensityVariance()
    Var_y = x.convolve(Y_KERNEL).IntensityVariance()
    Var_xy = x.convolve(XY_KERNEL).IntensityVariance()

    return (Var_x, Var_y, Var_xy)

# Contraste
def RMS (x: DiscreteFunction):
    """
    Interpretation

    Donne une première évaluation du contraste de l'image

    bien si [45, 75]
    """

    return x.IntensityStandardDeviation()

def HistogramSpread(x: DiscreteFunction):
    """
    Interpretation

    Donne une première évaluation du contraste de l'image

    bien si [0.27, 0.6]
    """

    import numpy as np

    # Récupération pixels
    pixels = np.array([
        x[i, j]
        for j in range(x.height)
        for i in range(x.width)
    ])

    # Histogramme (256 niveaux)
    hist, _ = np.histogram(pixels, bins=256, range=(0, 256))

    # Histogramme cumulé
    cdf = np.cumsum(hist)
    total = cdf[-1]

    # Normalisation en %
    cdf_normalized = cdf / total

    # Trouver I25 et I75
    I25 = np.searchsorted(cdf_normalized, 0.25)
    I75 = np.searchsorted(cdf_normalized, 0.75)

    # Histogram Spread
    HS = (I75 - I25) / 255

    return HS

def CumulativeHistogramCorrelation(x: DiscreteFunction):
    """
    Interpretation

    Donne une bonne évaluation du contraste de l'image

    - [0, 1]
    - correct [0.7, 1]
    - proche de 1, l'image est contrastée
    """

    hist = [0] * 256

    for j in range(x.height):
        for i in range(x.width):
            value = int(x[i, j])
            if 0 <= value <= 255:
                hist[value] += 1

    cdf = [0] * 256
    cumulative = 0
    total_pixels = x.width * x.height

    for k in range(256):
        cumulative += hist[k]
        cdf[k] = cumulative / total_pixels  # normalisation [0,1]

    # CDF idéal (rampe linéaire)
    cdf_ideal = [k / 255 for k in range(256)]

    mean_real = sum(cdf) / 256
    mean_ideal = sum(cdf_ideal) / 256

    # Corrélation de Pearson
    numerator = 0
    var_real = 0
    var_ideal = 0

    for k in range(256):
        a = cdf[k] - mean_real
        b = cdf_ideal[k] - mean_ideal

        numerator += a * b
        var_real += a * a
        var_ideal += b * b

    denominator = sqrt(var_real * var_ideal)

    if denominator == 0:
        return 0

    return numerator / denominator

def SaltAndPepperDetection(x: DiscreteFunction, show_plot=False):
    hist = x.getHistorigram()

    if show_plot:
        plt.figure()
        plt.plot(range(0, 256), hist)
        plt.plot(range(0, 256), [x.MeanIntensity()]*256)
        plt.plot(range(0, 256), [x.MeanIntensity() + x.IntensityStandardDeviation()]*256)
        plt.plot(range(0, 256), [x.MeanIntensity() - x.IntensityStandardDeviation()]*256)
        plt.show()

    total = x.height * x.width

    p_pepper = hist[0] / total       # pixels noirs
    p_salt = hist[255] / total       # pixels blancs
    p_salt_and_pepper = p_pepper + p_salt   # probabilité globale

    return p_salt_and_pepper

# Flou Gaussien
def radial_profile(data: DiscreteFunction):
    import numpy as np
    y, x = np.indices((data.height, data.width))
    center = np.array((data.height, data.width)) // 2
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)

    # Aplatir le kernel
    kernel_flat = np.array(data.kernel).ravel()

    # Somme et comptage pour chaque rayon
    tbin = np.bincount(r.ravel(), kernel_flat)
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


def ExtractSpacialStandartDeviation(f: DiscreteFunction):
    from MathematicsMethods import FFT2
    from Mathematics import ComplexDiscreteFunction
    import numpy as np

    F = ComplexDiscreteFunction(FFT2(f.kernel))
    F.normalize()  # normalisation

    S = F.getModule(False)
    profile = radial_profile(S)

    r = np.arange(len(profile))

    # éviter log(0)
    profile = np.maximum(profile, 1e-12)
    log_profile = np.log(profile)



    # Limiter le rayon pour éviter bruit
    r_max = len(profile) // 2
    mask = (r > 0) & (r < r_max)

    import matplotlib.pyplot as plt
    plt.plot(r[mask], log_profile[mask])
    plt.xlabel("Rayon")
    plt.ylabel("log(Module FFT)")
    plt.show()

    coeff = np.polyfit(r[mask]**2, log_profile[mask], 1)
    slope = coeff[0]
    print("Slope:", slope)

    # sigma estimation
    sigma_estimated = np.sqrt(-slope / (2 * np.pi**2))
    return sigma_estimated


############## Anciens #####################

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





def PartialAnalysis(source: DiscreteFunction, image: DiscreteFunction):
    """
    Analyzes the image compared to the source and determines which filter to apply based on indicators.
    
    Returns a string indicating the recommended filter.
    """
    
    # Compute PSNR to assess overall quality degradation
    psnr = PSNR(source, image)
    
    if psnr > 40:  # High quality, minimal degradation
        return "No filter needed"
    
    # Check for salt and pepper noise
    p_sp = SaltAndPepperDetection(image)
    if p_sp > 0.01:  # Significant salt and pepper noise detected
        return "Median filter"
    
    # Check for blur: compare Laplacian variance
    lap_source = LaplacianVariance(source)
    lap_image = LaplacianVariance(image)
    if lap_image < lap_source * 0.8:  # Image is significantly blurrier
        return "Sharpening filter"  # Note: Sharpening may need implementation
    
    # Check for general noise: compare variances
    var_source = source.IntensityVariance()
    var_image = image.IntensityVariance()
    if var_image > var_source * 1.2:  # Increased variance indicates noise
        return "Bilateral filter"
    
    # Check for low contrast
    rms_image = RMS(image)
    if rms_image < 50:  # Low contrast detected
        return "Histogram equalization"  # Note: May need implementation
    
    # Default case: assume Gaussian noise
    return ""
    

