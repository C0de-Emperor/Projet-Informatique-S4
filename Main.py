import Tests
#import Analysis, Noising
from Mathematics import *

#Tests.AdaptativeGaussianFilterTest(r'Pictures/Garden_strawberry.jpg')

#Tests.MedianFilterTest("Pictures/Garden_strawberry.jpg")

#Tests.FourierTransformTest("Pictures/toto.png", 10)

#Tests.RadiusCutTest("Pictures/baboon.png", 200, centered=True)

#Tests.InverseFourierTransformTest("Pictures/petites_fraises.png")

#Tests.AnalysisAdaptativeGaussianTest("Pictures/Garden_strawberry.jpg", -50, 50, 0.9, 80)

#Tests.FFTRadiusCutTest("Pictures/toto.png", 0.3, 50)

#if __name__=="__main__": Tests.FTsTimeTest(1, 1001, 100)

#Tests.AnalysisSaltAndPaperCurveVSMedian("Pictures/Garden_strawberry.jpg", p_max=1, steps=60)

#Tests.AnalysisRandomNoisingCurveVSGaussian("Pictures/Garden_strawberry.jpg", amplitude= 40, steps=60)

#f = DiscreteFunctionFromImage(r'Pictures/Garden_strawberry.jpg')
#source = f.copy()
#g = GaussianDiscreteFunction(2)
"""
f = DiscreteFunctionFromImage(r'Pictures/Garden_strawberry.jpg')
source = f.copy()
g = GaussianDiscreteFunction(2)

#h = f.convolve(g)
#h.show()

#f.apply(Noising.GaussianNoising, 0.22)
#f.show()


#result = Analysis.PartialAnalysis(source, f)

#print(result)

#print((a[0] - b[0], a[1] - b[1], a[2] - b[2]))


Tests.test_couleur("Pictures/petites_fraises.png")
"""

#f = DiscreteFunctionFromImage(r'Pictures\photo_Projet_2 (1).png')

#g = GaussianDiscreteFunction(2)

#f.show()

if __name__=="__main__":
    Tests.deconv_wiener("Pictures/mer.jpg")
