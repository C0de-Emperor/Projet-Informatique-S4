import Tests
import Analysis, Noising
from Mathematics import *

#Tests.AdaptativeGaussianFilterTest(r'Pictures\Garden_strawberry.jpg')

#Tests.MedianFilterTest("Pictures/Garden_strawberry.jpg")

#Tests.FourierTransformTest("Pictures/toto.png", 10)

#Tests.RadiusCutTest("Pictures/baboon.png", 200, centered=True)

#Tests.InverseFourierTransformTest("Pictures/petites_fraises.png")

#Tests.AnalysisAdaptativeGaussianTest("Pictures/Garden_strawberry.jpg", -50, 50, 0.9, 80)

#Tests.FFTRadiusCutTest("Pictures/toto.png", 0.3, 50)

#if __name__=="__main__": Tests.FTsTimeTest(1, 1001, 100)

#Tests.AnalysisSaltAndPaperCurveVSMedian("Pictures/Garden_strawberry.jpg", p_max=1, steps=60)

#Tests.AnalysisRandomNoisingCurveVSGaussian("Pictures/Garden_strawberry.jpg", amplitude= 40, steps=60)

f = DiscreteFunctionFromImage(r'Pictures\Garden_strawberry.jpg')
Analysis.SaltAndPepperDetection(f)
f.apply(Noising.saltAndPaperNoising, 0.2)

Analysis.SaltAndPepperDetection(f)
