import Tests
import Analysis
from Mathematics import *

#Tests.AdaptativeGaussianFilterTest(r'Pictures\Garden_strawberry.jpg')

#Tests.MedianFilterTest("Pictures/Garden_strawberry.jpg")

#Tests.FourierTransformTest("Pictures/toto.png", 10)

#Tests.RadiusCutTest("Pictures/baboon.png", 200, centered=True)

#Tests.InverseFourierTransformTest("Pictures/petites_fraises.png")

#Tests.AnalysisAdaptativeGaussianTest("Pictures/Garden_strawberry.jpg", -50, 50, 0.9, 80)

#Tests.FFTRadiusCutTest("Pictures/toto.png", 0.3, 50)

#if __name__=="__main__": Tests.FTsTimeTest(1, 10, 1)

#Tests.AnalysisSaltAndPaperCurveVSMedian("Pictures/Garden_strawberry.jpg", p_max=1, steps=60)

#Tests.AnalysisRandomNoisingCurveVSGaussian("Pictures/Garden_strawberry.jpg", amplitude= 40, steps=60)

f = DiscreteFunctionFromImage("Pictures/toto.png")

print(Analysis.SobelVariance(f))
