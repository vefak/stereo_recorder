#ifndef __stereocore_RT_H__
#define __stereocore_RT_H__

//#define WITH_GPU
//#define WITH_LANE_DETECTION
//#define WITH_OCCUPANCY_GRID
//#define WITH_DOCK_DETECTION
//#define WITH_SLAM
//#define WITH_SLAM_TEST
#define WITH_BUMBLEBEE
#define WITH_BUMBLEBEE_REALTIME



#include "iostream"
#include <iomanip>
#include <ctime>
#include <memory>
#include <fstream>
#include <sys/stat.h>
 

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>



#ifdef WITH_GPU
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#endif

#ifdef WITH_LANE_DETECTION
#include <opencv2/objdetect.hpp>
#define K_VARY_FACTOR 0.2f
#define B_VARY_FACTOR 20
#define MAX_LOST_FRAMES 30
#endif


#ifdef WITH_BUMBLEBEE_REALTIME
#include <triclops.h>
#include <fc2triclops.h>
namespace FC2 = FlyCapture2;
namespace FC2T = Fc2Triclops;
#endif

#ifdef WITH_GPU
using namespace cv::cuda;
#endif

using namespace cv;
using namespace std;


static int numAllocate = 0;
static int numDeallocate = 0;

template <class Tp>
struct NAlloc {
	typedef Tp value_type;
	NAlloc() = default;
	template <class T> NAlloc(const NAlloc<T>&) {}
	Tp* allocate(std::size_t n) {
		//numAllocate++;
		n *= sizeof(Tp);
		//std::cout << "allocating " << n << " bytes\n";
		//cout << "Allocate:" << numAllocate << endl;
		return static_cast<Tp*>(::operator new(n));
	}
	void deallocate(Tp* p, std::size_t n) {
		//numDeallocate++;
		//cout << "Deallocate:" << numDeallocate << endl;
		//std::cout << "deallocating " << n*sizeof*p << " bytes\n";
		::operator delete(p);
	}
};
template <class T, class U>
bool operator==(const NAlloc<T>&, const NAlloc<U>&) { return true; }
template <class T, class U>
bool operator!=(const NAlloc<T>&, const NAlloc<U>&) { return false; }


class StereoImageBuffer
{
	class vDisparityParams
	{
	public:
		int shiftedStartRangeD;
		int shiftedStartRangeV;

		unsigned int totalRows; //total rows of the vDisparity image
		unsigned int totalCols; //total columns of the vDisparity image
		unsigned int middleRow;

		double houghLinesP_rho;
		double houghLinesP_theta;
		int houghLinesP_threshold;
		double houghLinesP_minLineLength;
		double houghLinesP_maxLineGap;

		cv::Size gaussianBlur_size;
		double gaussianBlur_sigmaX;
		double gaussianBlur_sigmaY;
		cv::BorderTypes  gaussianBlur_borderType;

		cv::ThresholdTypes thresholding_type;
		double  thresholding_threshold;
		double  thresholding_maxVal;

		float bestLine_minDegreeThreshold;//linear equation min degree
		float bestLine_maxDegreeThreshold;//linear equation max degree
		float bestLine_minLenLengthThreshold;//linear min length

		float bestLineCurve_minDegreeThreshold;//min line degree for curve points of the second order polynom, if linear equation does not exist
		float bestLineCurve_maxDegreeThreshold;//max line degree for curve points of the second order polynom, if linear equation does not exist
		float bestLineCurve_minLenLengthThreshold;//min line length for curve points of the second order polynom, if linear equation does not exist
		float bestLineCurve_leftShiftD;//min disparity start point to avoid irrelevant houghLines
		float bestLineCurve_bottomShiftV;//max v point to avoid irrelevant houghLines

		int horizonLineV;

		vDisparityParams(int tRows, int tCols)
		{
			totalRows = tRows;
			totalCols = tCols;
			middleRow = round(totalRows / 2);
			horizonLineV = 450;// 170; //450(bumblebee)


			shiftedStartRangeD = 2; /*=0 if not blind spot*/
			shiftedStartRangeV = horizonLineV; // middleRow - (middleRow / 4); //middleRow; //Let's search in the half space (or under the horizon line) to find the bestline faster

			houghLinesP_threshold = 30;
			houghLinesP_minLineLength = 10;
			houghLinesP_maxLineGap = 20;
			houghLinesP_rho = 1;
			houghLinesP_theta = 1 * (CV_PI / 180);

			gaussianBlur_size = cv::Size(3, 5);
			gaussianBlur_sigmaX = 0;
			gaussianBlur_sigmaY = 0;
			gaussianBlur_borderType = cv::BorderTypes::BORDER_ISOLATED;

			thresholding_type = cv::ThresholdTypes::THRESH_OTSU;
			thresholding_threshold = 0;
			thresholding_maxVal = 255;

			//bestLine_minDegreeThreshold = 0.1;// tan function returns -89 for leftsided and 89 for rightsided lines, so we used abs()
			//bestLine_maxDegreeThreshold = 81.9;//// tan function returns -89 for leftsided and 89 for rightsided lines, so we used abs()


			bestLine_minDegreeThreshold = 45;
			bestLine_maxDegreeThreshold = 85;
			bestLine_minLenLengthThreshold = tCols*(1.5);

			bestLineCurve_minDegreeThreshold = 45;
			bestLineCurve_maxDegreeThreshold = 90;
			bestLineCurve_minLenLengthThreshold = tCols;
			bestLineCurve_leftShiftD = 5;
			bestLineCurve_bottomShiftV = horizonLineV*1.5;
		}

	};

	class uDisparityParams
	{
	public:
		int shiftedStartRangeD;
		unsigned short int maxDOfSideBuildings;//parameter when merging side lines, to avoid breaking side buildings lines
		unsigned short int minThicknessOfSideBuildings;//parameter when merging side lines
		unsigned short int minLengthOfSideBuildings;//parameter when merging side lines
		unsigned int totalRows; //total rows of the uDisparity image
		unsigned int totalCols; //total columns of the uDisparity image
		unsigned int middleCol; //middleCol of the uDisparity image
		unsigned int mainFrameLeftCol; //hypotetical main Frame Left Col of the uDisparity image
		unsigned int mainFrameRightCol; //hypotetical main Frame Right Col of the uDisparity image

		cv::Size gaussianBlur_size;
		double gaussianBlur_sigmaX;
		double gaussianBlur_sigmaY;
		cv::BorderTypes  gaussianBlur_borderType;

		cv::ThresholdTypes thresholding_type;
		cv::AdaptiveThresholdTypes thresholding_adaptiveMethod;
		double  thresholding_maxVal;
		double thresholding_SubtractionConstant_C;
		int thresholding_blockSize;

		//Params for labeling algorithm
		int zeroD;
		int maxD;
		int secondQuarterD;
		int firstQuarterD;
		int thirdQuarterD;
		//vertical masks (mask size height)
		int zeroD_MaskV;
		int maxD_MaskV;
		int threequarterD_MaskV;
		int halfD_MaskV;
		int quarterD_MaskV;

		int mainFrame_zeroD_MaskV;
		int mainFrame_maxD_MaskV;
		int mainFrame_threequarterD_MaskV;
		int mainFrame_halfD_MaskV;
		int mainFrame_quarterD_MaskV;
		//horizontal masks (mask size length)
		int zeroD_MaskH;
		int maxD_MaskH;
		int threequarterD_MaskH;
		int halfD_MaskH;
		int quarterD_MaskH;

		float zeroD_InclinedObject_AnglePrecision;
		float maxD_InclinedObject_AnglePrecision;
		float threequarterD_InclinedObject_AnglePrecision;
		float halfD_InclinedObject_AnglePrecision;
		float quarterD_InclinedObject_AnglePrecision;

		uDisparityParams(int tRows, int tCols)
		{
			shiftedStartRangeD = 2; /*Cropped from the top since lower d values are error prone*/
			minThicknessOfSideBuildings = 1;
			minLengthOfSideBuildings = tCols / 8;
			totalRows = tRows;
			totalCols = tCols;
			middleCol = round(tCols / 2);
			mainFrameLeftCol = middleCol - (round(tCols / 8) + round(tCols / 16));
			mainFrameRightCol = middleCol + (round(tCols / 8) + round(tCols / 16));

			gaussianBlur_size = cv::Size(3, 3);
			gaussianBlur_sigmaX = 0;
			gaussianBlur_sigmaY = 0;
			gaussianBlur_borderType = cv::BorderTypes::BORDER_ISOLATED;

			thresholding_type = cv::ThresholdTypes::THRESH_BINARY;
			thresholding_adaptiveMethod = cv::AdaptiveThresholdTypes::ADAPTIVE_THRESH_GAUSSIAN_C;
			thresholding_maxVal = 255;
			thresholding_SubtractionConstant_C = -11; //could be positive (usually) or negative
			thresholding_blockSize = 9;
			//if (shiftedStartRangeD > 0)	{ thresholding_blockSize = 5; } //could be 3,5,7... and so on
			//else{ thresholding_blockSize = 7; } //could be 3,5,7... and so on	

			maxD = totalRows;
			zeroD = 7;
			secondQuarterD = ((maxD + 1) / 2);
			firstQuarterD = ((maxD + 1) / 4);
			thirdQuarterD = (((maxD + 1) / 4) * 3);

			maxDOfSideBuildings = zeroD + 4;//maxD starting point of the side buildings

											//vertical masks (mask size height)
											//maxD_MaskV = 11;// total_rows / 4;
											//threequarterD_MaskV = 8;// (total_rows / 16) * 2;
											//halfD_MaskV = 5;// (total_rows / 32) * 3;
											//quarterD_MaskV = 3;// total_rows / 16;
											//zeroD_MaskV = 2;

											//bumblebee
			maxD_MaskV = 5;// total_rows / 4;
			threequarterD_MaskV = 4;// (total_rows / 16) * 2;
			halfD_MaskV = 3;// (total_rows / 32) * 3;
			quarterD_MaskV = 2;// total_rows / 16;
			zeroD_MaskV = 1;

			//mainFrame_maxD_MaskV = 10;// total_rows / 4;
			//mainFrame_threequarterD_MaskV = 7;// (total_rows / 16) * 2;
			//mainFrame_halfD_MaskV = 4;// (total_rows / 32) * 3;
			//mainFrame_quarterD_MaskV = 3;// total_rows / 16;
			//mainFrame_zeroD_MaskV = 2;

			//bumblebee
			mainFrame_maxD_MaskV = 6;// total_rows / 4;
			mainFrame_threequarterD_MaskV = 5;// (total_rows / 16) * 2;
			mainFrame_halfD_MaskV = 4;// (total_rows / 32) * 3;
			mainFrame_quarterD_MaskV = 3;// total_rows / 16;
			mainFrame_zeroD_MaskV = 2;

			//horizontal masks (mask size length)
			maxD_MaskH = maxD_MaskV * 3;// 14;// (totalCols / 128) * 3;
			threequarterD_MaskH = threequarterD_MaskV * 3;// 12;// totalCols / 64;
			halfD_MaskH = halfD_MaskV * 3;// 10;// (totalCols / 256) * 3;
			quarterD_MaskH = quarterD_MaskV * 3;// 8;// totalCols / 128;
			zeroD_MaskH = zeroD_MaskV * 3;//6;// totalCols / 128;

										  //bumblebee
										  //maxD_MaskH = maxD_MaskV * 2;// 14;// (totalCols / 128) * 3;
										  //threequarterD_MaskH = threequarterD_MaskV * 2;// 12;// totalCols / 64;
										  //halfD_MaskH = halfD_MaskV * 2;// 10;// (totalCols / 256) * 3;
										  //quarterD_MaskH = quarterD_MaskV * 2;// 8;// totalCols / 128;
										  //zeroD_MaskH = zeroD_MaskV * 2;//6;// totalCols / 128;

			zeroD_InclinedObject_AnglePrecision = 1.5;
			maxD_InclinedObject_AnglePrecision = 6;
			threequarterD_InclinedObject_AnglePrecision = 4.5;
			halfD_InclinedObject_AnglePrecision = 3.5;
			quarterD_InclinedObject_AnglePrecision = 2.5;
		}
	};

	class uDisparityGlobals
	{
		int Global_maxNumOfObjectAllowed;
	public:
		cv::Mat uDisparityImage; //created uDisparityImage
		cv::Mat uDisparityImage2; //created uDisparityImage
		cv::Mat uDisparityImage_shifted; //gaussian blurred uDisparityImage
		cv::Mat uDisparityImage_blurred; //gaussian blurred uDisparityImage
		cv::Mat uDisparityImage_thresholded; //thresholded uDisparityImage
		cv::Mat uDisparityImage_labelled; //labelled uDisparityImage by Labeling Alg.

		int sideLineLeftIndex;
		int sideLineRightIndex;

		int infinityCommonPointX;
		int infinityCommonPointY;
		int infinityLeftPointX;
		int infinityLeftPointY;
		int infinityRightPointX;
		int infinityRightPointY;

		float sideLineLeft_m;
		float sideLineLeft_b;
		float sideLineRight_m;
		float sideLineRight_b;

		uDisparityGlobals()
		{
			//cout << "Creating uDisparityGlobals" << endl;
			sideLineLeftIndex = -1;
			sideLineRightIndex = -1;
			infinityCommonPointX = -1;
			infinityCommonPointY = -1;
			infinityLeftPointX = -1;
			infinityLeftPointY = -1;
			infinityRightPointX = -1;
			infinityRightPointY = -1;
			sideLineLeft_m = 0;
			sideLineLeft_b = 0;
			sideLineRight_m = 0;
			sideLineRight_b = 0;
			Global_maxNumOfObjectAllowed = numeric_limits<unsigned char>::max(); //so maxLabel == 255
		}

		~uDisparityGlobals()
		{
			//cout << "Deleting uDisparityGlobals" << endl;
		}

		void clear()
		{
			sideLineLeftIndex = -1;
			sideLineRightIndex = -1;
			infinityCommonPointX = -1;
			infinityCommonPointY = -1;
			sideLineLeft_m = 0;
			sideLineLeft_b = 0;
			sideLineRight_m = 0;
			sideLineRight_b = 0;
		}
	};

	class vDisparityGlobals
	{
	public:
		unsigned int bestRoadLine_vIntersection; //v intersection of the bestline (Labayrade, 2002)
		unsigned int bestRoadLine_dInterSection; //g intersection of the bestline (recheck!!!)
		float bestRoadLine_m; //m*x param of the bestline (y = m*x + a*x^2 + b)
		float bestRoadLine_b; //+ b param of the bestline (y = m*x + a*x^2 + b)
		float bestRoadLine_a; //a*x^2 param of the bestline (y = m*x + a*x^2 + b)

		cv::Vec4i bestRoadLine; //the bestline vector
		cv::Mat vDisparityImage; //created vDisparityImage
		cv::Mat vDisparityImage_Baseline; //Real Baseline points of disparites computed from world to Image coordinates
		cv::Mat vDisparityImage2; //created vDisparityImage
		cv::Mat vDisparityImage_shifted; //gaussian blurred vDisparityImage
		cv::Mat vDisparityImage_blurred; //gaussian blurred vDisparityImage
		cv::Mat vDisparityImage_thresholded; //thresholded vDisparityImage
		std::vector<cv::Vec4i> allLines; //lines vectors found by the hough transform
		cv::Vec4i piecewiseBestLine;//piecewise best line vector
		cv::Point2i piecewiseBestLine_RoadEndPoint;
		cv::Point2i piecewiseBestLine_RoadStartPoint;

		vDisparityGlobals()
		{
			//cout << "Creating vDisparityGlobals" << endl;
			bestRoadLine_m = 0;
			bestRoadLine_b = 0;
			bestRoadLine_a = 0;
			bestRoadLine_dInterSection = 0;
			bestRoadLine_vIntersection = 0;
		}

		~vDisparityGlobals()
		{
			//cout << "Deleting vDisparityGlobals" << endl;
		}

		void clear()
		{
			bestRoadLine_m = 0;
			bestRoadLine_b = 0;
			bestRoadLine_a = 0;
			bestRoadLine_dInterSection = 0;
			bestRoadLine_vIntersection = 0;
		}
	};

	class dDisparityParams
	{
	public:
		unsigned int totalRows; //total rows of the vDisparity image
		unsigned int totalCols; //total columns of the vDisparity image
		unsigned int middleCol;
		float upperline_extendDisparitySearchConst;
		int minVminVmaxDifDeleteThreshold;
		int upperline_minVminVmaxDifThreshold;
		int upperline_minCounterThresholdConst;
		float minObjSizeAllowed;
		float maxObjSizeAllowed;
		float minObjHeightAllowed;

		dDisparityParams(int tRows, int tCols)
		{
			totalRows = tRows;
			totalCols = tCols;
			upperline_extendDisparitySearchConst = 0.18;//stop if it reaches the %x of the bottomline of the object's length 
			minVminVmaxDifDeleteThreshold = 2;//delete object if (make it 2)
			upperline_minCounterThresholdConst = 8;
			upperline_minVminVmaxDifThreshold = 30;
			middleCol = round(tCols / 2);
			minObjSizeAllowed = 0.25; //meter square
			maxObjSizeAllowed = 10.0;
			minObjHeightAllowed = 0.1;//delete objects smaller than 10cm
		}
	};

	class dDisparityGlobals
	{
	public:
		int infinityPointX;
		int infinityPointY;
		int infinityUlineCommonPointX;
		int infinityUlineCommonPointY;
		int infinityUlineLeftPointX;
		int infinityUlineLeftPointY;
		int infinityUlineRightPointX;
		int infinityUlineRightPointY;
		float vehicleRoadLineLeft_m;
		float vehicleRoadLineLeft_b;
		float vehicleRoadLineRight_m;
		float vehicleRoadLineRight_b;

		dDisparityGlobals()
		{
			infinityPointX = 0;
			infinityPointY = 0;
			infinityUlineCommonPointX = -1;
			infinityUlineCommonPointY = -1;
			infinityUlineLeftPointX = -1;
			infinityUlineLeftPointY = -1;
			infinityUlineRightPointX = -1;
			infinityUlineRightPointY = -1;
			vehicleRoadLineLeft_m = 0;
			vehicleRoadLineLeft_b = 0;
			vehicleRoadLineRight_m = 0;
			vehicleRoadLineRight_b = 0;
		}

		~dDisparityGlobals()
		{
			//cout << "Deleting uDisparityGlobals" << endl;
		}

		void clear()
		{
			infinityPointX = -1;
			infinityPointY = -1;
			infinityUlineCommonPointX = -1;
			infinityUlineCommonPointY = -1;
			infinityUlineLeftPointX = -1;;
			infinityUlineLeftPointY = -1;;
			infinityUlineRightPointX = -1;;
			infinityUlineRightPointY = -1;;
			vehicleRoadLineLeft_m = 0;
			vehicleRoadLineLeft_b = 0;
			vehicleRoadLineRight_m = 0;
			vehicleRoadLineRight_b = 0;
		}
	};

#ifdef WITH_BUMBLEBEE
	class Bumblebee
	{
	public:
		int outputWidth = 1280;// = 1280 // 1152 // 1024 // 960 // 800 // 768 // 640 // 512 // 400 // 384 // 320;
		int outputHeight = 960;// = 960 //  864  // 768 //  720 // 600 // 576 // 480 // 384 // 300 // 288 // 240;

		FC2T::StereoCameraMode cameraStereoMode;//Camere operating mode

		TriclopsCameraConfiguration cameraStereoCalculationMode;//Stereo Calculation parameter reference

		enum resolutions
		{
			Res_1280X960 = 1,
			Res_1152X864,
			Res_1024X768,
			Res_960X720,
			Res_800X600,
			Res_768X576,
			Res_640X480,
			Res_512X384,
			Res_400X300,
			Res_384X288,
			Res_320X240,
		} resolution;

		struct opencvParams
		{
			int blockSize = 19;//(blockSize) default 15
			int minDisp = 1;// 0;
			int numDisp = 128;// 240;

			TriclopsStereoAlgorithm selectedAlgorithm_LeftREF = TriAlg_OCVSGBMLR;//Left Reference disparity;
			TriclopsStereoAlgorithm selectedAlgorithm_RightREF = TriAlg_OCVSGBM; //Right Reference disparity		

			int                 prefilterSize = 11;//11 
			int                 prefilterCap = 11;//11 def:31
			OpenCVPrefilterType prefilterType = OCV_XSOBEL;

			int textureThreshold = 1;//BM
			int speckleSize = 0;//1 def:200;
			int speckleRange = 0; //0 def:4;
			int uniquenessRatio = 1; //1 def:10;

			int            SGBM_P1 = blockSize * blockSize * 8;
			int            SGBM_P2 = blockSize * blockSize * 32;
			OpenCVSGBMMode SGBM_Mode = OCV_SGBM;//5 ways default //OCV_SGBM_3WAY
		} SGBM_Params;

		Bumblebee()
		{

		}

		Bumblebee(bool wideMode, resolutions res, int numDisp, int blockSize, bool use3way)
		{
			SGBM_Params.blockSize = blockSize;
			SGBM_Params.numDisp = numDisp;
			SGBM_Params.SGBM_P1 = SGBM_Params.blockSize * SGBM_Params.blockSize * 8;
			SGBM_Params.SGBM_P2 = SGBM_Params.blockSize * SGBM_Params.blockSize * 32;

			if (use3way)
				SGBM_Params.SGBM_Mode = OCV_SGBM_3WAY;

			if (wideMode)
			{
				cameraStereoMode = FC2T::StereoCameraMode::TWO_CAMERA_WIDE;

				cameraStereoCalculationMode = TriclopsCameraConfiguration::TriCfg_2CAM_HORIZONTAL_WIDE;

				stereoParams.b = 0.239987001;
				stereoParams.cR = 0.500507;
				stereoParams.cC = 0.498539;

				switch (res)
				{
				case resolutions::Res_1280X960:
				{
					outputWidth = 1280; outputHeight = 960;
					stereoParams.f = 960.129272;
					stereoParams.u0 = 640.148926;
					stereoParams.v0 = 478.097443;
					break;
				}

				case resolutions::Res_1152X864:
				{
					outputWidth = 1152; outputHeight = 864;
					stereoParams.f = 864.116333;
					stereoParams.u0 = 576.084045;
					stereoParams.v0 = 430.237701;
					break;
				}

				case resolutions::Res_1024X768:
				{
					outputWidth = 1024; outputHeight = 768;
					stereoParams.f = 768.103394;
					stereoParams.u0 = 512.019165;
					stereoParams.v0 = 382.377960;
					break;
				}

				case resolutions::Res_960X720:
				{
					outputWidth = 960; outputHeight = 720;
					stereoParams.f = 720.096924;
					stereoParams.u0 = 479.986755;
					stereoParams.v0 = 358.448090;
					break;
				}

				case resolutions::Res_800X600:
				{
					outputWidth = 800; outputHeight = 600;
					stereoParams.f = 600.080750;
					stereoParams.u0 = 399.905579;
					stereoParams.v0 = 298.623413;
					break;
				}

				case resolutions::Res_768X576:
				{
					outputWidth = 768; outputHeight = 576;
					stereoParams.f = 576.077515;
					stereoParams.u0 = 383.889374;
					stereoParams.v0 = 286.658478;
					break;
				}

				case resolutions::Res_640X480:
				{
					outputWidth = 640; outputHeight = 480;
					stereoParams.f = 480.064636;
					stereoParams.u0 = 319.824463;
					stereoParams.v0 = 238.798721;
					break;
				}

				case resolutions::Res_512X384:
				{
					outputWidth = 512; outputHeight = 384;
					stereoParams.f = 384.051697;
					stereoParams.u0 = 255.759583;
					stereoParams.v0 = 190.938980;
					break;
				}

				case resolutions::Res_400X300:
				{
					outputWidth = 400; outputHeight = 300;
					stereoParams.f = 300.040375;
					stereoParams.u0 = 199.702805;
					stereoParams.v0 = 149.061707;
					break;
				}

				case resolutions::Res_384X288:
				{
					outputWidth = 384; outputHeight = 288;
					stereoParams.f = 288.038757;
					stereoParams.u0 = 191.694687;
					stereoParams.v0 = 143.079224;
					break;
				}

				case resolutions::Res_320X240:
				{
					outputWidth = 320; outputHeight = 240;
					stereoParams.f = 240.032318;
					stereoParams.u0 = 159.662247;
					stereoParams.v0 = 119.149361;
					break;
				}
				}
			}
			else//if Narrow Mode
			{
				cameraStereoMode = FC2T::StereoCameraMode::TWO_CAMERA_NARROW;

				cameraStereoCalculationMode = TriclopsCameraConfiguration::TriCfg_2CAM_HORIZONTAL_NARROW;

				stereoParams.b = 0.120008998;
				stereoParams.cR = 0.501731;
				stereoParams.cC = 0.503426;

				switch (res)
				{
				case resolutions::Res_1280X960:
				{
					outputWidth = 1280; outputHeight = 960;
					stereoParams.f = 959.722229;
					stereoParams.u0 = 641.715698;
					stereoParams.v0 = 482.788971;
					break;
				}

				case resolutions::Res_1152X864:
				{
					outputWidth = 1152; outputHeight = 864;
					stereoParams.f = 863.750000;
					stereoParams.u0 = 577.494141;
					stereoParams.v0 = 434.460052;
					break;
				}

				case resolutions::Res_1024X768:
				{
					outputWidth = 1024; outputHeight = 768;
					stereoParams.f = 767.777771;
					stereoParams.u0 = 513.272583;
					stereoParams.v0 = 386.131165;
					break;
				}

				case resolutions::Res_960X720:
				{
					outputWidth = 960; outputHeight = 720;
					stereoParams.f = 719.791687;
					stereoParams.u0 = 481.161774;
					stereoParams.v0 = 361.966705;
					break;
				}

				case resolutions::Res_800X600:
				{
					outputWidth = 800; outputHeight = 600;
					stereoParams.f = 599.826355;
					stereoParams.u0 = 400.884796;
					stereoParams.v0 = 301.555603;
					break;
				}

				case resolutions::Res_768X576:
				{
					outputWidth = 768; outputHeight = 576;
					stereoParams.f = 575.833313;
					stereoParams.u0 = 384.829407;
					stereoParams.v0 = 289.473389;
					break;
				}

				case resolutions::Res_640X480:
				{
					outputWidth = 640; outputHeight = 480;
					stereoParams.f = 479.861115;
					stereoParams.u0 = 320.607849;
					stereoParams.v0 = 241.144485;
					break;
				}

				case resolutions::Res_512X384:
				{
					outputWidth = 512; outputHeight = 384;
					stereoParams.f = 383.888885;
					stereoParams.u0 = 256.386261;
					stereoParams.v0 = 192.815582;
					break;
				}

				case resolutions::Res_400X300:
				{
					outputWidth = 400; outputHeight = 300;
					stereoParams.f = 299.913177;
					stereoParams.u0 = 200.192398;
					stereoParams.v0 = 150.527802;
					break;
				}

				case resolutions::Res_384X288:
				{
					outputWidth = 384; outputHeight = 288;
					stereoParams.f = 287.916656;
					stereoParams.u0 = 192.164703;
					stereoParams.v0 = 144.486694;
					break;
				}

				case resolutions::Res_320X240:
				{
					outputWidth = 320; outputHeight = 240;
					stereoParams.f = 239.930557;
					stereoParams.u0 = 160.053925;
					stereoParams.v0 = 120.322243;
					break;
				}
				}
			}

		}

		struct stereoCam_Params
		{
			//parameters Cam
			float f = 960.129272; // focal length[pixel] NARROW:959.722229 - WIDE:960.129272
			float sx = 1.0; // pixel width X [no unit]
			float sy = 1.0; // pixel width Y [no unit]
			float u0 = 640.149048; // principal point X [pixel] NARROW:641.715698 - WIDE:640.149048
			float v0 = 478.097443;// // principal point Y [pixel] NARROW:482.788971 - WIDE:478.097443
			float b = 0.239987001; // baseline [meter] NARROW:0.120008998 - WIDE:0.239987001
			float cX = 0.0; // camera lateral position [meter]
			float cY = 0.0; // camera height above ground [meter]
			float cZ = 0.0; // camera offset Z [meter]
			float tilt = 0.0; //camera tilt angle [rad] (NOTE: should be adjusted by online tilt angle estimation)
			float cR = 0.0;//Pinhole center Row
			float cC = 0.0;//Pinhole center Col
		} stereoParams;


		//const struct stereoCam_NARROW_Params
		//{
		//	//parameters Cam
		//	float f = 959.722229; // focal length[pixel] NARROW:959.722229 - WIDE:960.129272
		//	float sx = 1.0; // pixel width X [no unit]
		//	float sy = 1.0; // pixel width Y [no unit]
		//	float u0 = 641.715698; // principal point X [pixel] NARROW:641.715698 - WIDE:640.149048
		//	float v0 = 482.788971;// // principal point Y [pixel] NARROW:482.788971 - WIDE:478.097443
		//	float b = 0.120008998; // baseline [meter] NARROW:0.120008998 - WIDE:0.239987001
		//	float cX = 0.0; // camera lateral position [meter]
		//	float cY = 0.0; // camera height above ground [meter]
		//	float cZ = 0.0; // camera offset Z [meter]
		//	float tilt = 0.0; //camera tilt angle [rad] (NOTE: should be adjusted by online tilt angle estimation)
		//} stereoNARROWParams_1280X960;	



		bool setUpDefaults()
		{

		}
	};

	struct ImageContainer
	{
		FC2::Image tmp[2];
		FC2::Image unprocessed[2];
	};

	enum IMAGE_SIDE
	{
		RIGHT = 0, LEFT
	};

	void generatemonoStereoPair(TriclopsContext const &context, FC2::Image const &grabbedImage, ImageContainer &imageCont, TriclopsMonoStereoPair &stereoPair)
	{
		FC2T::ErrorType fc2TriclopsError;
		TriclopsError te;

		TriclopsImage triclopsImageContainer[2];
		FC2::Image *tmpImage = imageCont.tmp;
		FC2::Image *unprocessedImage = imageCont.unprocessed;

		// Convert the pixel interleaved raw data to de-interleaved and color processed data
		fc2TriclopsError = FC2T::unpackUnprocessedRawOrMono16Image(grabbedImage, true /*assume little endian*/, tmpImage[RIGHT], tmpImage[LEFT]);

		//handleError("FC2T::unpackUnprocessedRawOrMono16Image()", fc2TriclopsError, __LINE__);
		unprocessedImage[RIGHT] = tmpImage[RIGHT];
		unprocessedImage[LEFT] = tmpImage[LEFT];

		// create triclops image for right and left lens
		for (size_t i = 0; i < 2; ++i) {
			te = triclopsLoadImageFromBuffer(unprocessedImage[i].GetData(), unprocessedImage[i].GetRows(), unprocessedImage[i].GetCols(), unprocessedImage[i].GetStride(), &triclopsImageContainer[i]);
			//triclops_examples::handleError("triclopsLoadImageFromBuffer()", te, __LINE__);
		}

		// create stereo input from the triclops images constructed above
		// pack image data into a TriclopsColorStereoPair structure
		te = triclopsBuildMonoStereoPairFromBuffers(context, &triclopsImageContainer[RIGHT], &triclopsImageContainer[LEFT], &stereoPair);
		//triclops_examples::handleError("triclopsBuildMonoStereoPairFromBuffers()", te, __LINE__);
	}

	// convert a triclopsImage mono image to opencv mat
	Mat convertTriclops2OpencvMat(TriclopsImage& disparityImage)
	{
		Mat cvImage;
		cvImage = cv::Mat(disparityImage.nrows, disparityImage.ncols, CV_8UC1, disparityImage.data, disparityImage.rowinc);
		return cvImage;
	}

	// convert a triclopsImage16 disparity image to opencv mat
	Mat convertTriclops2OpencvMat(TriclopsImage16& inputImage)
	{
		Mat cvImage;
		cvImage = cv::Mat(inputImage.nrows, inputImage.ncols, CV_16UC1, inputImage.data, inputImage.rowinc);
		return cvImage;
	}
#endif

#ifdef WITH_OCCUPANCY_GRID

	class occupancyGridParams
	{

	public:
		//uDisp occupancy Params
		float uDisp_probabilityFP;//False Positive Probability for uDisp occupancy
		float uDisp_probabilityFN;//False Negative Probability for uDisp occupancy
		float AprioriCellOccupancy;
		float uDisp_tO;//Constant for exponential function for Obstacles for uDisp occupancy
		float uDisp_tR;//Constant for exponential function for the Road for uDisp occupancy
		float uDisp_probabilityThreshold;//%50 means it is by chance for uDisp occupancy
		int uDisp_uResolution;//udisp surface x coordinate resolution (number of u to use to find left and right coord. of the surface)
		int uDisp_uHalfResolution;
		int uDisp_dResolution;

		//occupancy Grid Matrix Params
		float XWorldLimit; //x coordinate limit in meters, will be multipled by 2 (since there is left and right sides)
		float ZWorldLimit; //depth limit in meters
		float XWorldResolution; //cells per meter
		float ZWorldResolution; //cells per meter

		float oneSideWidth;
		float width;//map width in meters (x coordinate, limitX * 2), XWorldLimits
		float height;//map height in meters (depth), YWorldLimits
		float resolutionWidth;//width resolution, cells per meter
		float resolutionHeight;//depth resolution, cells per meter

		int totalCols;//occupancyMatrix total num of cols with respect to Height and Height resolution, GridSize
		int middleCol, middleRow;
		int totalRows;//occupancyMatrix total num of rows with respect to Width and Width resolution, GridSize

#ifdef WITH_SLAM
		int totalColsSLAM;//occupancyMatrix SLAM Map total num of cols with respect to Height and Height resolution, GridSize
		int totalRowsSLAM;//occupancyMatrix SLAM Map total num of cols with respect to Height and Height resolution, GridSize
		int middleColSLAM;//occupancyMatrix SLAM Map total num of cols with respect to Height and Height resolution, GridSize
		int middleRowSLAM;//occupancyMatrix SLAM Map total num of cols with respect to Height and Height resolution, GridSize
		bool useBilinearInterpolation;
		bool useJointProbilities;

		//define rotate points
		Point2f centerSLAM;
		Point2f rotateOriginOccupancyGrid;
		float xVelocity, zVelocity, yawDegree, fps, spf;
		Mat M, MI;

#endif

		occupancyGridParams()
		{
			uDisp_probabilityFP = 0.01;
			uDisp_probabilityFN = 0.05;
			AprioriCellOccupancy = 0.500000;//default uniform distribution, should be updated by the result of the previous step
			uDisp_probabilityThreshold = 0.5;
			uDisp_tO = 0.25;//0.1; --> smaller the ratio increase the object detection but also false the positives
			uDisp_tR = 0.1;

			XWorldLimit = 10.0;//20m lateral
			ZWorldLimit = 15.0;//40m depth
			XWorldResolution = 0.05; //0.05;//10cm
			ZWorldResolution = 0.05; //0.05;//10cm

			oneSideWidth = XWorldLimit;
			width = XWorldLimit * 2;
			height = ZWorldLimit;
			resolutionWidth = XWorldResolution;
			resolutionHeight = ZWorldResolution;

			totalCols = round(width / resolutionWidth);
			totalRows = round(height / resolutionHeight);
			middleCol = (int)round(totalCols / 2);
			middleRow = (int)round(totalRows / 2);

			uDisp_uResolution = 2;
			uDisp_uHalfResolution = uDisp_uResolution / 2;
			uDisp_dResolution = 1;

#ifdef WITH_SLAM
			//totalColsSLAM = (int)round(totalCols*((360.0 / 66.0)));
			totalColsSLAM = (int)round(totalCols * 2);
			middleColSLAM = (int)round((float)totalColsSLAM / 2.0);
			totalRowsSLAM = totalRows * 2;
			middleRowSLAM = (int)round((float)totalRowsSLAM / 2.0);
			useBilinearInterpolation = true;
			useJointProbilities = false;

			//totalColsSLAM = totalCols;
			//middleColSLAM = middleCol;
			//totalRowsSLAM = totalRows;
			//middleRowSLAM = middleRow;

			//define rotate points
			rotateOriginOccupancyGrid.x = middleCol;//rotate origin x
			rotateOriginOccupancyGrid.y = totalRows;//rotate origin y
			centerSLAM.x = middleColSLAM;//tx
			centerSLAM.y = middleRowSLAM;//ty

			xVelocity = 0.0;
			zVelocity = 0.0; // m/s = 10 km/h;
			yawDegree = 0.0;
			fps = 37;//frame per second
			spf = 37 / 60;//second per frame
#endif

		}


#ifdef WITH_SLAM
		void setRotationMatrix()
		{
			float zTrans = this->zVelocity / this->ZWorldResolution;
			float xTrans = this->xVelocity / this->XWorldResolution;

			//MI
			//z = -7.70  --> x*sin(teta) - z*cos(teta)
			//x = -38.91 --> -x*cos(teta) - z*sin(teta)

			//M
			//z = 38.91 -->  z*sin(teta) + x*cos(teta)
			//x = 7.70 -->   z*cos(teta) - x*sin(teta)

			M = cv::Mat::zeros(2, 3, CV_32FC1);
			M.at<float>(0, 0) = cos(this->yawDegree * CV_PI / 180.0);//cos(teta)
			M.at<float>(0, 1) = sin(this->yawDegree * CV_PI / 180.0);//sin(teta)
			M.at<float>(0, 2) = xTrans;// zTrans*sin(this->yawDegree * CV_PI / 180.0) + xTrans*cos(this->yawDegree * CV_PI / 180.0);// xTrans;// this->rotateOriginOccupancyGrid.y;//y trans
			M.at<float>(1, 0) = -sin(this->yawDegree * CV_PI / 180.0);//-sin(teta)
			M.at<float>(1, 1) = cos(this->yawDegree * CV_PI / 180.0);//cosd(teta)
			M.at<float>(1, 2) = zTrans;// zTrans*cos(this->yawDegree * CV_PI / 180.0) - xTrans*sin(this->yawDegree * CV_PI / 180.0);// zTrans;// this->rotateOriginOccupancyGrid.x;//x trans

			Mat MI2 = cv::Mat::zeros(2, 3, CV_32FC1);
			cv::invertAffineTransform(M, MI2);

			//M.at<float>(0, 2) = zTrans*sin(this->yawDegree * CV_PI / 180.0) + xTrans*cos(this->yawDegree * CV_PI / 180.0);//xtrans
			//M.at<float>(1, 2) = zTrans*cos(this->yawDegree * CV_PI / 180.0) - xTrans*sin(this->yawDegree * CV_PI / 180.0);//ztrans

			MI = cv::Mat::zeros(2, 3, CV_32FC1);
			MI.at<float>(0, 0) = cos(this->yawDegree * CV_PI / 180.0);//cos(teta)
			MI.at<float>(0, 1) = -sin(this->yawDegree * CV_PI / 180.0);//sin(teta)
																	   //MI.at<float>(0, 2) = zTrans*sin(this->yawDegree * CV_PI / 180.0) - xTrans*cos(this->yawDegree * CV_PI / 180.0);// zTrans;// this->rotateOriginOccupancyGrid.x;//x trans
			MI.at<float>(1, 0) = sin(this->yawDegree * CV_PI / 180.0);//-sin(teta)
			MI.at<float>(1, 1) = cos(this->yawDegree * CV_PI / 180.0);//cosd(teta)
																	  //MI.at<float>(1, 2) = -zTrans*cos(this->yawDegree * CV_PI / 180.0) - xTrans*sin(this->yawDegree * CV_PI / 180.0);  // xTrans;// this->rotateOriginOccupancyGrid.y;//y trans//

			MI.at<float>(0, 2) = -xTrans*sin(this->yawDegree * CV_PI / 180.0) + zTrans*cos(this->yawDegree * CV_PI / 180.0);// zTrans;// this->rotateOriginOccupancyGrid.x;//x trans
			MI.at<float>(1, 2) = xTrans*cos(this->yawDegree * CV_PI / 180.0) + zTrans*sin(this->yawDegree * CV_PI / 180.0);

			//cout << "M" << endl;

			//for (int r = 0; r < M.rows; r++)
			//{
			//	for (int c = 0; c < M.cols; c++)
			//	{
			//		cout << fixed << std::setprecision(2) << M.at<float>(r, c) << "  ";
			//	}
			//	cout << endl;
			//}

			//cout << "MI" << endl;

			//for (int r = 0; r < MI.rows; r++)
			//{
			//	for (int c = 0; c < MI.cols; c++)
			//	{
			//		cout << fixed << std::setprecision(2) << MI.at<float>(r, c) << "  ";
			//	}
			//	cout << endl;
			//}

			//cout << "MI2" << endl;

			//for (int r = 0; r < MI.rows; r++)
			//{
			//	for (int c = 0; c < MI.cols; c++)
			//	{
			//		cout << fixed << std::setprecision(2) << MI2.at<float>(r, c) << "  ";
			//	}
			//	cout << endl;
			//}

			//cout << endl;
		}

		Point2i setMapCoordinatesByNN(float x, float y)
		{
			float ct = (x - this->rotateOriginOccupancyGrid.x);// (c - origin_x)
			float rt = (y - this->rotateOriginOccupancyGrid.y);//(r - origin_y)

			x = ct*M.at<float>(0, 0) + rt*M.at<float>(0, 1) + this->centerSLAM.x + M.at<float>(0, 2);
			y = ct*M.at<float>(1, 0) + rt*M.at<float>(1, 1) + this->centerSLAM.y + M.at<float>(1, 2);

			x = round(x);// NN interpolation
			y = round(y);// NN interpolation

			int xi = (int)x;
			int yi = (int)y;

			if (xi < 0)
				xi = 0;

			if (xi >= this->totalColsSLAM)
				xi = this->totalColsSLAM - 1;

			if (yi < 0)
				yi = 0;

			if (yi >= this->totalRowsSLAM)
				yi = this->totalRowsSLAM - 1;

			Point2i xy;
			xy.x = xi;
			xy.y = yi;

			return xy;
		}
#endif


		~occupancyGridParams()
		{

		}
	};

	class uOccupancyCell
	{
	public:
		float leftU, rightU, downD, upD;//surface coordinates on uDisparity
		float P;//occupancy probability of the cell
		int d, u; //the integer disparity to calculate occupancy

		float surfaceLeftUpX, surfaceLeftDownX, surfaceRightUpX, surfaceRightDownX; //surface 3D coordinates, Point.x = Xw
		float surfaceLeftUpZ, surfaceLeftDownZ, surfaceRightUpZ, surfaceRightDownZ; //surface 3D coordinates, Point.y = Zw (depth)
		bool isLeftSided;//side of the object information
		float upLength, downLength, leftHeight, rightHeight, depthHeight;
		float leftLine_m, leftLine_b, rightLine_m, rightLine_b;
		int index;
		bool surfaceBoundaryUpdated;
		bool isValid;
		float PVu, PCu;
		vector<Point> gridPoints_InfluencedByTheSurface;
#ifdef WITH_SLAM
		float posteriorP;
		float aprioriP;//previous occupancy probability of the cell: default 0.5
		float xVelocity, zVelocity, yawDegree;
		int mapIndex;
		vector<Point> mapPoints_InfluencedByTheSurface;
#endif

		uOccupancyCell(float visibility, float confidence, int dCell, int uCell, float uLeft, float uRight, float dUp, float dDown, int cellIndex)
		{
			PVu = visibility;
			PCu = confidence;
			d = dCell;
			u = uCell;
			index = cellIndex;
			leftU = uLeft;
			rightU = uRight;
			upD = dUp;
			downD = dDown;
			isLeftSided = false;
			surfaceBoundaryUpdated = false;
			isValid = false;

#ifdef WITH_SLAM
			aprioriP = 0.5;//aPrioriP is %50 %50
			xVelocity = 0;
			zVelocity = 0;
			yawDegree = 0;
#endif

			ImageToWorldCoordinates(*this);

			upLength = abs(surfaceLeftUpX - surfaceRightUpX);
			downLength = abs(surfaceLeftDownX - surfaceRightDownX);
			depthHeight = surfaceLeftUpZ - surfaceLeftDownZ;
			leftHeight = sqrt(pow(depthHeight, 2) + pow(surfaceLeftUpX - surfaceLeftDownX, 2));
			rightHeight = sqrt(pow(depthHeight, 2) + pow(surfaceRightUpX - surfaceRightDownX, 2));

			leftLine_m = (surfaceLeftUpZ - surfaceLeftDownZ) / (surfaceLeftUpX - surfaceLeftDownX);
			rightLine_m = (surfaceRightUpZ - surfaceRightDownZ) / (surfaceRightUpX - surfaceRightDownX);

			leftLine_b = surfaceLeftUpZ - leftLine_m*surfaceLeftUpX;
			rightLine_b = surfaceRightUpZ - rightLine_m*surfaceRightUpX;

			if ((this->surfaceLeftDownX + this->surfaceRightDownX) < 0)//if the bigger side of the object is on the left side then it's leftsided
			{
				isLeftSided = true;
			}
		}

		void setOccupancy(float jointP)
		{
			P = jointP;
		}

#ifdef WITH_SLAM
		void setPosteriorOccupancy(float jointP)
		{
			posteriorP = jointP;
		}


		void setMapIndex()//should be defined according to x and z velocity of the cell (also yawRate)
		{
			mapIndex = index;
		}
#endif
		//void setMapCoordinates(int& x, int &z, float yawDegree, float vx, float vz)//should be defined according to x and z velocity of the cell (also yawRate)
		//{
		//	x = x;//+ xVelocity
		//	z = z;//+ zVelocity
		//	//* yawrate
		//	// there should be the transform matrix with x and z (y) translation + rotation degree accordimt to the yaw rate
		//}

		void setGridPoints(occupancyGridParams& param)
		{
			if (this->surfaceLeftDownZ < param.height) //if closest side of the surface is in the ZWorldLimit (this->surfaceLeftDown.y = this->surfaceRightDown.y)
			{
				if (this->surfaceLeftUpZ > param.height)
				{
					this->surfaceLeftUpZ = param.height;
					this->surfaceRightUpZ = param.height;
					this->surfaceBoundaryUpdated = true;
				}

				if ((this->isLeftSided && (abs(this->surfaceRightDownX) < param.oneSideWidth)) || ((!this->isLeftSided) && (abs(this->surfaceLeftDownX) < param.oneSideWidth)))//if closest side of the x coordinates of the surface is in the XWorldLimit
				{
					if ((this->isLeftSided && (abs(this->surfaceLeftDownX) > param.oneSideWidth)))
					{
						this->surfaceLeftDownX = -param.oneSideWidth;

						if (abs(this->surfaceLeftUpX) > param.oneSideWidth)
						{
							this->surfaceLeftUpX = -param.oneSideWidth;

							if (abs(this->surfaceRightUpX) > param.oneSideWidth)
							{
								this->surfaceRightUpX = -param.oneSideWidth + param.XWorldResolution;
							}
						}

						this->surfaceBoundaryUpdated = true;
					}
					else
					{
						if ((!this->isLeftSided) && (abs(this->surfaceRightDownX) > param.oneSideWidth))
						{
							this->surfaceRightDownX = param.oneSideWidth;

							if (abs(this->surfaceRightUpX) > param.oneSideWidth)
							{
								this->surfaceRightUpX = param.oneSideWidth;

								if (abs(this->surfaceLeftUpX) > param.oneSideWidth)
								{
									this->surfaceLeftUpX = param.oneSideWidth - param.XWorldResolution;
								}
							}

							this->surfaceBoundaryUpdated = true;
						}
					}

					this->isValid = true;
				}
			}

			if (this->isValid)
			{
				//float surfaceLeftUpX = StereoImageBuffer::my_round(uCell.surfaceLeftUpX, 1, true);
				float surfaceLeftUpZ = StereoImageBuffer::my_round(this->surfaceLeftUpZ, 1, true);
				//float surfaceRightUpX = StereoImageBuffer::my_round(uCell.surfaceRightUpX, 1, true);
				//float surfaceRightUpZ = StereoImageBuffer::my_round(uCell.surfaceRightUpZ, 1, true);

				//float surfaceLeftDownX = StereoImageBuffer::my_round(uCell.surfaceLeftDownX, 1, true);
				float surfaceLeftDownZ = StereoImageBuffer::my_round(this->surfaceLeftDownZ, 1, true);
				//float surfaceRightDownX = StereoImageBuffer::my_round(uCell.surfaceRightDownX, 1, true);
				//float surfaceRightDownZ = StereoImageBuffer::my_round(uCell.surfaceRightDownZ, 1, true);

				float h = (surfaceLeftUpZ - surfaceLeftDownZ);
				int maxNumOfPointsVertical = my_round(h / param.ZWorldResolution, 0, true) + 1;

				//float test_leftUpX = (surfaceLeftUpZ - uCell.leftLine_b) / uCell.leftLine_m;
				//test_leftUpX = StereoImageBuffer::my_round(test_leftUpX, 1, true);

				//float test_leftDownX = (surfaceLeftDownZ - uCell.leftLine_b) / uCell.leftLine_m;
				//test_leftDownX = StereoImageBuffer::my_round(test_leftDownX, 1, true);

				//float test_rightUpX = (surfaceRightUpZ - uCell.rightLine_b) / uCell.rightLine_m;
				//test_rightUpX = StereoImageBuffer::my_round(test_rightUpX, 1, true);

				//float test_rightDownX = (surfaceRightDownZ - uCell.rightLine_b) / uCell.rightLine_m;
				//test_rightDownX = StereoImageBuffer::my_round(test_rightDownX, 1, true);

				//cout << "(LU_X,LU_Z): (" << uCell.surfaceLeftUpX << "," << uCell.surfaceLeftUpZ << ")" << endl;
				//cout << "(LD_X,LD_Z): (" << uCell.surfaceLeftDownX << "," << uCell.surfaceLeftDownZ << ")" << endl;
				//cout << "(RU_X,RU_Z): (" << uCell.surfaceRightUpX << "," << uCell.surfaceRightUpZ << ")" << endl;
				//cout << "(RD_X,RD_Z): (" << uCell.surfaceRightDownX << "," << uCell.surfaceRightDownZ << ")" << endl << endl;

				//cout << "(LU_X,LU_Z): (" << surfaceLeftUpX << "," << surfaceLeftUpZ << ") X:" << test_leftUpX << endl;
				//cout << "(LD_X,LD_Z): (" << surfaceLeftDownX << "," << surfaceLeftDownZ << ") X:" << test_leftDownX << endl;
				//cout << "(RU_X,RU_Z): (" << surfaceRightUpX << "," << surfaceRightUpZ << ") X:" << test_rightUpX << endl;
				//cout << "(RD_X,RD_Z): (" << surfaceRightDownX << "," << surfaceRightDownZ << ") X:" << test_rightDownX << endl << endl;

				float startZ = surfaceLeftDownZ;
				float currentZ = startZ;
				float currentLeftLineIntersection_XStart, currentRightLineIntersection_XEnd;// , currentRightLineIntersection_XStart, currentLeftLineIntersection_XMiddle, currentRightLineIntersection_XMiddle, currentLeftLineIntersection_XEnd;
				for (int gridZIndex = 1; gridZIndex < maxNumOfPointsVertical; gridZIndex++)
				{
					currentLeftLineIntersection_XStart = (currentZ - this->leftLine_b) / this->leftLine_m;
					//currentRightLineIntersection_XStart = (currentZ - uCell.rightLine_b) / uCell.rightLine_m;

					//currentLeftLineIntersection_XMiddle = (currentZ + (param.ZWorldResolution / 2) - uCell.leftLine_b) / uCell.leftLine_m;
					//currentRightLineIntersection_XMiddle = (currentZ + (param.ZWorldResolution / 2) - uCell.rightLine_b) / uCell.rightLine_m;

					//currentLeftLineIntersection_XEnd = (currentZ + param.ZWorldResolution - uCell.leftLine_b) / uCell.leftLine_m;
					currentRightLineIntersection_XEnd = (currentZ + param.ZWorldResolution - this->rightLine_b) / this->rightLine_m;

					//cout << "Z1:" << currentZ << " --> (LX,RX): (" << currentLeftLineIntersection_XStart << "," << currentRightLineIntersection_XStart << ") --> (" << my_round(currentLeftLineIntersection_XStart, 1, true) << "," << my_round(currentRightLineIntersection_XStart, 1, true) << ")" << endl;
					//cout << "Z2:" << currentZ + (param.ZWorldResolution / 2) << " --> (LX,RX): (" << currentLeftLineIntersection_XMiddle << "," << currentRightLineIntersection_XMiddle << ") --> (" << my_round(currentLeftLineIntersection_XMiddle, 1, true) << "," << my_round(currentRightLineIntersection_XMiddle, 1, true) << ")" << endl;
					//cout << "Z3:" << currentZ + param.ZWorldResolution << " --> (LX,RX): (" << currentLeftLineIntersection_XEnd << "," << currentRightLineIntersection_XEnd << ") --> (" << my_round(currentLeftLineIntersection_XEnd, 1, true) << "," << my_round(currentRightLineIntersection_XEnd, 1, true) << ")" << endl;

					float leftFloor_XStart = floor(currentLeftLineIntersection_XStart * 10) / 10;
					//float leftCeil_XStart = ceil(currentLeftLineIntersection_XStart * 10) / 10;
					//float leftMiddle_XStart = (leftFloor_XStart + leftCeil_XStart) / 2;

					//float leftFloor_XMiddle = floor(currentLeftLineIntersection_XMiddle*10) / 10;
					//float leftCeil_XMiddle = ceil(currentLeftLineIntersection_XMiddle*10) / 10;
					//float leftMiddle_XMiddle = (leftFloor_XMiddle + leftCeil_XMiddle) / 2;

					//float leftFloor_XEnd = floor(currentLeftLineIntersection_XEnd * 10) / 10;
					//float leftCeil_XEnd = ceil(currentLeftLineIntersection_XEnd * 10) / 10;
					//float leftMiddle_XEnd = (leftFloor_XEnd + leftCeil_XEnd) / 2;

					//float rightFloor_XMiddle = floor(currentRightLineIntersection_XMiddle * 10) / 10;
					//float rightCeil_XMiddle = ceil(currentRightLineIntersection_XMiddle * 10) / 10;
					//float rightMiddle_XMiddle = (rightFloor_XMiddle + rightCeil_XMiddle) / 2;

					//float rightFloor_XStart = floor(currentRightLineIntersection_XStart * 10) / 10;
					//float rightCeil_XStart = ceil(currentRightLineIntersection_XStart * 10) / 10;
					//float rightMiddle_XStart = (rightFloor_XStart + rightCeil_XStart) / 2;

					//float rightFloor_XEnd = floor(currentRightLineIntersection_XEnd * 10) / 10;
					float rightCeil_XEnd = ceil(currentRightLineIntersection_XEnd * 10) / 10;
					//float rightMiddle_XEnd = (rightFloor_XEnd + rightCeil_XEnd) / 2;

					//cout << "Left--> Floor:" << leftFloor_XStart << " X:" << currentLeftLineIntersection_XStart << " Ceil:" << leftCeil_XStart << " Middle:" << leftMiddle_XStart << endl;
					//cout << "Right--> Floor:" << rightFloor_XStart << " X:" << currentRightLineIntersection_XStart << " Ceil:" << rightCeil_XStart << " Middle:" << rightMiddle_XStart << endl;

					//cout << "Left--> Floor:" << leftFloor_XMiddle << " X:" << currentLeftLineIntersection_XMiddle << " Ceil:" << leftCeil_XMiddle << " Middle:" << leftMiddle_XMiddle << endl;
					//cout << "Right--> Floor:" << rightFloor_XMiddle << " X:" << currentRightLineIntersection_XMiddle << " Ceil:" << rightCeil_XMiddle << " Middle:" << rightMiddle_XMiddle << endl;

					//cout << "Left--> Floor:" << leftFloor_XEnd << " X:" << currentLeftLineIntersection_XEnd << " Ceil:" << leftCeil_XEnd << " Middle:" << leftMiddle_XEnd << endl;
					//cout << "Right--> Floor:" << rightFloor_XEnd << " X:" << currentRightLineIntersection_XEnd << " Ceil:" << rightCeil_XEnd << " Middle:" << rightMiddle_XEnd << endl << endl;

					int startX;
					//if (currentLeftLineIntersection_XStart > 0)
					//{
					startX = (leftFloor_XStart / param.XWorldResolution) + param.middleCol;
					//}
					//else
					//{
					//	startX = (leftCeil_XStart / param.XWorldResolution) + middleCol;
					//}

					int endX;
					//if (currentRightLineIntersection_XEnd > 0)
					//{
					endX = (rightCeil_XEnd / param.XWorldResolution) + param.middleCol;
					//}
					//else
					//{
					//	endX = (rightFloor_XEnd / param.XWorldResolution) + middleCol;
					//}

					int z = (int)round(param.totalRows - (currentZ / param.ZWorldResolution) - 1);

					for (int x = startX + 1; x <= endX; x++)
					{
						this->gridPoints_InfluencedByTheSurface.push_back(Point(x, z));
#ifdef WITH_SLAM
						if (!(param.useBilinearInterpolation))
						{
							cv::Point2i new_xy = param.setMapCoordinatesByNN((float)x, (float)z);
							this->mapPoints_InfluencedByTheSurface.push_back(Point(new_xy.x, new_xy.y));
						}
#endif
						//cout << " Add (x,z): (" << x << "," << z << ")" << endl;
					}

					//cout << endl;

					currentZ = currentZ + param.ZWorldResolution;

					//cout << floor(-2.69*10)/10 << " " << ceil(-2.69*10)/10 << endl;				
				}//for
			}
		}//function
	};

	class occupancyGrid
	{
	public:
		Mat occupancyMatrix;
		Mat occupancyMatrix_Thresholded;

#ifdef WITH_SLAM
		Mat occupancyMatrixSLAM;
		Mat occupancyMatrixSLAM_Thresholded;
		Mat occupancyMatrixPCu;
		Mat occupancyMatrixPVu;
#endif		

		occupancyGrid()
		{
		}

		/*		void print_OccupancyGridMatrix()
		{
		std::ostringstream text;
		text << "[";

		Mat temp = this->occupancyMatrix;

		for (int gridRowIndex = 0; gridRowIndex < this->totalRows; gridRowIndex++)
		{
		for (int gridColIndex = 0; gridColIndex < this->totalCols; gridColIndex++)
		{
		float value = this->occupancyMatrix.at<float>(gridRowIndex, gridColIndex);
		text << fixed << std::setprecision(2) << value << " ";
		}
		text << ";";
		}


		ofstream myfile;
		myfile.open("occupancy.txt");
		myfile << text.str();
		myfile.close();

		text << "]";
		text.str("");
		text.clear();
		}
		*/

		void set_OccupancyGridMatrix(occupancyGridParams& param)
		{
			//Mat occupancyMatrix_init(cv::Size(param.totalCols, param.totalRows), CV_32FC1, Scalar(param.AprioriCellOccupancy));
			//occupancyMatrix = occupancyMatrix_init;

			occupancyMatrix = cv::Mat(cv::Size(param.totalCols, param.totalRows), CV_32FC1, Scalar(param.AprioriCellOccupancy));

			//occupancyMatrix = cv::Mat::create( (cv::Size(param.totalRows, param.totalCols), CV_32FC1, Scalar(param.AprioriCellOccupancy));
			//occupancyMatrix = Mat::zeros(param.totalRows, param.totalCols, CV_32FC1);
			occupancyMatrix_Thresholded = cv::Mat::zeros(param.totalRows, param.totalCols, CV_32FC1);
		}

#ifdef WITH_SLAM
		void set_OccupancyGridSLAMMatrix(occupancyGridParams& param)
		{

			//occupancyMatrixSLAM = Mat::zeros(param.totalRowsSLAM, param.totalColsSLAM, CV_32FC1);
			occupancyMatrixSLAM = cv::Mat(cv::Size(param.totalColsSLAM, param.totalRowsSLAM), CV_32FC1, Scalar(param.AprioriCellOccupancy));
			occupancyMatrixSLAM_Thresholded = cv::Mat(cv::Size(param.totalColsSLAM, param.totalRowsSLAM), CV_8UC1, Scalar(100));

			Mat oc = occupancyMatrixSLAM_Thresholded;

			if (param.useJointProbilities)
			{
				occupancyMatrixPCu = Mat::zeros(param.totalRows, param.totalCols, CV_32FC1);
				occupancyMatrixPVu = Mat::zeros(param.totalRows, param.totalCols, CV_32FC1);
			}
		}

		void set_OccupancyGridSLAMMatrix_ByBilinearInterpolation(occupancyGridParams& param)
		{
			float x_real, y_real;
			int x, y;
			float ct, rt;
			float gridDefaultValue = 0;

			//find boundaries not to search all pixels to interpolate
			float *xPoints = new float[4];
			float *yPoints = new float[4];

			//[0,0]
			float rectLeftUpPointX = 0, rectLeftUpPointX_map;
			float rectLeftUpPointY = 0, rectLeftUpPointY_map;
			ct = (rectLeftUpPointX - param.rotateOriginOccupancyGrid.x);// (c - origin_x)
			rt = (rectLeftUpPointY - param.rotateOriginOccupancyGrid.y);//(r - origin_y)
			rectLeftUpPointX_map = ct*param.M.at<float>(0, 0) + rt*param.M.at<float>(0, 1) + param.centerSLAM.x + param.M.at<float>(0, 2);
			rectLeftUpPointY_map = ct*param.M.at<float>(1, 0) + rt*param.M.at<float>(1, 1) + param.centerSLAM.y + param.M.at<float>(1, 2);
			xPoints[0] = rectLeftUpPointX_map;
			yPoints[0] = rectLeftUpPointY_map;

			//[0,c]
			float rectRightUpPointX = param.totalCols, rectRightUpPointX_map;
			float rectRightUpPointY = 0, rectRightUpPointY_map;
			ct = (rectRightUpPointX - param.rotateOriginOccupancyGrid.x);// (c - origin_x)
			rt = (rectRightUpPointY - param.rotateOriginOccupancyGrid.y);//(r - origin_y)
			rectRightUpPointX_map = ct*param.M.at<float>(0, 0) + rt*param.M.at<float>(0, 1) + param.centerSLAM.x + param.M.at<float>(0, 2);
			rectRightUpPointY_map = ct*param.M.at<float>(1, 0) + rt*param.M.at<float>(1, 1) + param.centerSLAM.y + param.M.at<float>(1, 2);
			xPoints[1] = rectRightUpPointX_map;
			yPoints[1] = rectRightUpPointY_map;

			//[r,0]
			float rectLeftDownPointX = 0, rectLeftDownPointX_map;
			float rectLeftDownPointY = param.totalRows, rectLeftDownPointY_map;
			ct = (rectLeftDownPointX - param.rotateOriginOccupancyGrid.x);// (c - origin_x)
			rt = (rectLeftDownPointY - param.rotateOriginOccupancyGrid.y);//(r - origin_y)
			rectLeftDownPointX_map = ct*param.M.at<float>(0, 0) + rt*param.M.at<float>(0, 1) + param.centerSLAM.x + param.M.at<float>(0, 2);
			rectLeftDownPointY_map = ct*param.M.at<float>(1, 0) + rt*param.M.at<float>(1, 1) + param.centerSLAM.y + param.M.at<float>(1, 2);
			xPoints[2] = rectLeftDownPointX_map;
			yPoints[2] = rectLeftDownPointY_map;

			//[r,c]
			float rectRightDownPointX = param.totalCols, rectRightDownPointX_map;
			float rectRightDownPointY = param.totalRows, rectRightDownPointY_map;
			ct = (rectRightDownPointX - param.rotateOriginOccupancyGrid.x);// (c - origin_x)
			rt = (rectRightDownPointY - param.rotateOriginOccupancyGrid.y);//(r - origin_y)
			rectRightDownPointX_map = ct*param.M.at<float>(0, 0) + rt*param.M.at<float>(0, 1) + param.centerSLAM.x + param.M.at<float>(0, 2);
			rectRightDownPointY_map = ct*param.M.at<float>(1, 0) + rt*param.M.at<float>(1, 1) + param.centerSLAM.y + param.M.at<float>(1, 2);
			xPoints[3] = rectRightDownPointX_map;
			yPoints[3] = rectRightDownPointY_map;

			//find the smallest and biggest x and y points to draw rectangle boundaries to search for interpolation

			//for rows (x Points)
			float rStart = param.totalRowsSLAM;
			float rEnd = 0;
			for (int ry = 0; ry < 4; ry++)
			{
				if (yPoints[ry] < rStart)
				{
					rStart = yPoints[ry];
				}

				if (yPoints[ry] > rEnd)
				{
					rEnd = yPoints[ry];
				}
			}

			if (rStart < 0)
				rStart = 0;

			if (rEnd > param.totalRowsSLAM)
				rEnd = param.totalRowsSLAM;

			//for columns (y Points)
			float cStart = param.totalColsSLAM;
			float cEnd = 0;
			for (int rx = 0; rx < 4; rx++)
			{
				if (xPoints[rx] < cStart)
				{
					cStart = xPoints[rx];
				}

				if (xPoints[rx] > cEnd)
				{
					cEnd = xPoints[rx];
				}
			}

			if (cStart < 0)
				cStart = 0;

			if (cEnd > param.totalColsSLAM)
				cEnd = param.totalColsSLAM;

			int rStarti = (int)round(rStart);
			int cStarti = (int)round(cStart);
			int rEndi = (int)round(rEnd);
			int cEndi = (int)round(cEnd);
			//find boundaries not to search all pixels to interpolate

			for (int r = rStarti; r < rEndi; r++)
			{
				for (int c = cStarti; c < cEndi; c++)
				{
					//ct = (c - (param.centerSLAM.x + param.M.at<float>(0, 2)));// (c - origin_x)
					//rt = (r - (param.centerSLAM.y + param.M.at<float>(1, 2)));//(r - origin_y)

					ct = (c - (param.centerSLAM.x + param.M.at<float>(0, 2)));// (c - origin_x)
					rt = (r - (param.centerSLAM.y + param.M.at<float>(1, 2)));//(r - origin_y)

					x_real = ct*param.MI.at<float>(0, 0) + rt*param.MI.at<float>(0, 1) + param.rotateOriginOccupancyGrid.x;// +param.MI.at<float>(0, 2);
					y_real = ct*param.MI.at<float>(1, 0) + rt*param.MI.at<float>(1, 1) + param.rotateOriginOccupancyGrid.y;// +param.MI.at<float>(1, 2);

					float x2 = y_real;
					float y2 = x_real;

					float u = floor(x2);
					float v = floor(y2);

					float W1 = (u + 1 - x2)*(v + 1 - y2);
					float W2 = (x2 - u)*(v + 1 - y2);
					float W3 = (u + 1 - x2)*(y2 - v);
					float W4 = (x2 - u)*(y2 - v);

					if ((u >= 0 && u < occupancyMatrix.rows - 1) && (v >= 0 && v < (occupancyMatrix.cols - 1)))
					{
						float resPOu = W1* occupancyMatrix.at<float>(u, v) + W2* occupancyMatrix.at<float>(u + 1, v) + W3* occupancyMatrix.at<float>(u, v + 1) + W4* occupancyMatrix.at<float>(u + 1, v + 1);//also previous POu (a-priori for joint probility)
						float previousSLAMValue = occupancyMatrixSLAM.at<float>(r, c);

						//if (previousSLAMValue == param.AprioriCellOccupancy && (resPOu > 0 && resPOu < 0.1))
						//{
						//	std::cout << "(" << r << "," << c << ") - W1:" << W1 << " u:" << u << " v:" << v << "(" << occupancyMatrix.at<float>(u, v) << ")" << " - W2:" << W2 << " (u+1):" << u + 1 << " v:" << v << "(" << occupancyMatrix.at<float>(u+1, v) << ")" << " - W3:" << W3 << " u:" << u << " (v+1):" << v + 1 << "(" << occupancyMatrix.at<float>(u, v+1) << ")" << " - W4:" << W4 << " (u+1):" << u + 1 << " (v+1):" << v + 1 << "(" << occupancyMatrix.at<float>(u+1, v+1) << "): " << resPOu  << std::endl;
						//}

						if (!param.useJointProbilities)
						{
							if (resPOu != param.AprioriCellOccupancy && (resPOu > previousSLAMValue || previousSLAMValue == param.AprioriCellOccupancy))//if not using joint probilities just use the argMax							
								occupancyMatrixSLAM.at<float>(r, c) = resPOu;

							//if (resPOu != param.AprioriCellOccupancy)//override (param.AprioriCellOccupancy == grid default value)
							//{
							//	occupancyMatrixSLAM.at<float>(r, c) = resPOu;
							//}
						}
						else
						{
							//if (resPOu != param.AprioriCellOccupancy) //since otherwise its not a calculated value, it's default value of the current occupancy grid
							//{
							float jointPou;
							if (/*previousSLAMValue == 0 ||*/ previousSLAMValue == param.AprioriCellOccupancy)//if it is default value or 0.50 then use predefined uniform a-priori (AprioriCellOccupancy) //resPOu = jointPou
							{
								//jointPou = resPVu*resPCu*(1 - param.uDisp_probabilityFP) + resPVu*(1 - resPCu)*param.uDisp_probabilityFN + (1 - resPVu)*param.AprioriCellOccupancy;
								jointPou = resPOu;
							}
							else
							{
								float resPCu = W1* occupancyMatrixPCu.at<float>(u, v) + W2* occupancyMatrixPCu.at<float>(u + 1, v) + W3* occupancyMatrixPCu.at<float>(u, v + 1) + W4* occupancyMatrixPCu.at<float>(u + 1, v + 1);
								float resPVu = W1* occupancyMatrixPVu.at<float>(u, v) + W2* occupancyMatrixPVu.at<float>(u + 1, v) + W3* occupancyMatrixPVu.at<float>(u, v + 1) + W4* occupancyMatrixPVu.at<float>(u + 1, v + 1);
								jointPou = resPVu*resPCu*(1 - param.uDisp_probabilityFP) + resPVu*(1 - resPCu)*param.uDisp_probabilityFN + (1 - resPVu)*previousSLAMValue;
							}

							occupancyMatrixSLAM.at<float>(r, c) = jointPou;

							if (jointPou > param.AprioriCellOccupancy)
								occupancyMatrixSLAM_Thresholded.at<unsigned char>(r, c) = 255;
							else
							{
								if (jointPou < param.AprioriCellOccupancy)
									occupancyMatrixSLAM_Thresholded.at<unsigned char>(r, c) = 0;
								else
									occupancyMatrixSLAM_Thresholded.at<unsigned char>(r, c) = 100;
							}

						}
					}
				}
			}

		}
#endif


		void world2grid(uOccupancyCell& uCell, occupancyGridParams& param)
		{
			Mat test_occupancyMatrix = this->occupancyMatrix;//delete later
			int countError = 0;

#ifdef WITH_SLAM
			//float anteriorValueSLAM = -1;//check if all grid cells has the same Pou to avoid redoing the POu calculation
#endif
			//world2gridIndex(uCell, param);

			if (uCell.isValid)
			{
				float POu;
				int total_size = uCell.gridPoints_InfluencedByTheSurface.size();

				POu = uCell.PVu*uCell.PCu*(1 - param.uDisp_probabilityFP) + uCell.PVu*(1 - uCell.PCu)*param.uDisp_probabilityFN + (1 - uCell.PVu)*param.AprioriCellOccupancy;
				uCell.setOccupancy(POu);

				for (int cellIndx = 0; cellIndx < total_size; cellIndx++)
				{
					Point cell = uCell.gridPoints_InfluencedByTheSurface.back();
#ifdef WITH_SLAM
					Point mapCell;
					if (!(param.useBilinearInterpolation))
					{
						mapCell = uCell.mapPoints_InfluencedByTheSurface.back();
					}
#endif

					bool discardFromSLAM = false;////since we don't want to add points to SLAM avoided in the worldToGrid function
					if (cell.y < param.totalRows && cell.x < param.totalCols && cell.y >= 0 && cell.x >= 0)
					{
						float previousValue = this->occupancyMatrix.at<float>(cell.y, cell.x);

						if (uCell.P > previousValue || previousValue == param.AprioriCellOccupancy)//argMax --> Perrolaz, if (previousValue == defValue) add even it is small to keep free space values, if uCell.P == 0 (not FP) then it is not a calculated value
						{
							this->occupancyMatrix.at<float>(cell.y, cell.x) = uCell.P;

#ifdef WITH_SLAM
							if (param.useJointProbilities)//then keep the argMax Pvu and PCu for probiblity density function
							{
								this->occupancyMatrixPCu.at<float>(cell.y, cell.x) = uCell.PCu;
								this->occupancyMatrixPVu.at<float>(cell.y, cell.x) = uCell.PVu;
							}

#endif
							if (uCell.P > 0.50)
								this->occupancyMatrix_Thresholded.at<float>(cell.y, cell.x) = uCell.P;
						}
					}
					else
					{
						discardFromSLAM = true;
					}

					uCell.gridPoints_InfluencedByTheSurface.pop_back();

#ifdef WITH_SLAM
					if (!(param.useBilinearInterpolation))
					{
						if (!discardFromSLAM && mapCell.y < param.totalRowsSLAM && mapCell.x < param.totalColsSLAM && mapCell.y >= 0 && mapCell.x >= 0)
						{
							float previousValueSLAM = this->occupancyMatrixSLAM.at<float>(mapCell.y, mapCell.x);

							if (uCell.P > previousValueSLAM || previousValueSLAM == param.AprioriCellOccupancy)
							{
								this->occupancyMatrixSLAM.at<float>(mapCell.y, mapCell.x) = uCell.P;
							}
							//else
							//{
							//	if (uCell.P > previousValueSLAM)//argMax --> Perrolaz
							//	{
							//		this->occupancyMatrixSLAM.at<float>(mapCell.y, mapCell.x) = uCell.P;
							//	}
							//}

							//if (previousValueSLAM == anteriorValueSLAM)
							//{
							//	this->occupancyMatrixSLAM.at<float>(mapCell.y, mapCell.x) = uCell.posteriorP;
							//	anteriorValueSLAM = previousValueSLAM;
							//}
							//else
							//{
							//	if (previousValueSLAM == 0)
							//	{
							//		POu = uCell.PVu*uCell.PCu*(1 - param.uDisp_probabilityFP) + uCell.PVu*(1 - uCell.PCu)*param.uDisp_probabilityFN + (1 - uCell.PVu)*param.AprioriCellOccupancy;// or uCell.aprioriP which is also set to .AprioriCellOccupancy
							//		uCell.setPosteriorOccupancy(POu);

							//		this->occupancyMatrixSLAM.at<float>(mapCell.y, mapCell.x) = uCell.posteriorP;
							//	}
							//	else
							//	{
							//		uCell.aprioriP = previousValueSLAM;//let's just keep the last value as the Pou of the CELL occupancy
							//		POu = uCell.PVu*uCell.PCu*(1 - param.uDisp_probabilityFP) + uCell.PVu*(1 - uCell.PCu)*param.uDisp_probabilityFN + (1 - uCell.PVu)*uCell.aprioriP;
							//		uCell.setPosteriorOccupancy(POu);

							//		//if (uCell.P - uCell.posteriorP > 0.10)
							//		//cout << uCell.index << "-" << uCell.posteriorP << " - " << uCell.P << uCell.index << endl;

							//		//if (uCell.posteriorP > previousValueSLAM)//argMax --> Perrolaz
							//		//{
							//			this->occupancyMatrixSLAM.at<float>(mapCell.y, mapCell.x) = uCell.posteriorP;
							//		//}
							//	}
							//}
						}
						else
						{
							//cout << "Index:" << umapCell.index << " (" << mapCell.x << "," << mapCell.y << ")" << " D: (" << umapCell.surfaceLeftDownX << "," << umapCell.surfaceRightDownX << "," << umapCell.surfaceRightDownZ << ")" << " U: (" << umapCell.surfaceLeftUpX << "," << umapCell.surfaceRightUpX << "," << umapCell.surfaceRightUpZ << ")" << " isnt added" << endl;
							//countError++;
						}

						uCell.mapPoints_InfluencedByTheSurface.pop_back();
					}//if not use bilinear Interpolation
#endif
				}
			}

			//if (uCell.index >= 5400 && uCell.index <= 5420)
			//	cout << " posteriorP: " << uCell.posteriorP << " *" << endl << endl;

			//test_occupancyMatrix = this->occupancyMatrix;
			//cout << "Errors:" << countError << endl;
		}

#ifdef WITH_SLAM
		void setMapPoints(float &x, float &y, occupancyGridParams& param)
		{



			//float rectRightUpPointX = Global_occupancyGridParams->totalCols, rectRightUpPointX_map;
			//			float rectRightUpPointY = 0, rectRightUpPointY_map;
			//			ct = (rectRightUpPointX - centerSLAM.x);// (c - origin_x)
			//			rt = (rectRightUpPointY - centerSLAM.y);//(r - origin_y)
			//			rectRightUpPointX_map = ct*M.at<double>(0, 0) + rt*M.at<double>(0, 1) + centerSLAM.x;
			//			rectRightUpPointY_map = ct*M.at<double>(1, 0) + rt*M.at<double>(1, 1) + centerSLAM.y;
		}
#endif
	};

	class occupancyGridGlobals
	{
	public:
		occupancyGrid occupancyGridMap;

#ifdef WITH_SLAM
		vector<shared_ptr<uOccupancyCell>> uDispOccupancyCells;
#endif

		occupancyGridGlobals()
		{

		}
		~occupancyGridGlobals()
		{

		}
	};
#endif

#ifdef WITH_LANE_DETECTION
	class laneDetection
	{
		cv::CascadeClassifier laneCascadeClassifier;

		cv::Size gaussianBlur_size;
		double gaussianBlur_sigmaX;
		double gaussianBlur_sigmaY;
		cv::BorderTypes  gaussianBlur_borderType;

		bool canny_L2gradient;
		int canny_threshold1;
		int canny_threshold2;
		int canny_aperturesize;

		double houghLinesP_rho;
		double houghLinesP_theta;
		int houghLinesP_threshold;
		double houghLinesP_minLineLength;
		double houghLinesP_maxLineGap;

		double laneScaleFactor;
		float lineMinRejectDegree;
		float lineMaxRejectDegree;
		int SCAN_STEP;			  // in pixels
		int BW_TRESHOLD;		  // edge response strength to recognize for 'WHITE'
		int BORDERX;			  // px, skip this much from left & right borders
		int MAX_RESPONSE_DIST;	  // px


		int halfWidth, halfHeight, startX;
		Mat src;

		struct Lane {
			Lane(Point a, Point b, float angle, float kl, float bl) : p0(a), p1(b), votes(0),
				visited(false), found(false), angle(angle), k(kl), b(bl) { }

			Point p0, p1;
			int votes;
			bool visited, found;
			float angle, k, b;
		};

		class ExpMovingAverage {
		private:
			double alpha; // [0;1] less = more stable, more = less stable
			double oldValue;
			bool unset;

		public:

			ExpMovingAverage() {
				this->alpha = 0.2;
				unset = true;
			}

			void clear() {
				unset = true;
			}

			void add(double value) {
				if (unset) {
					oldValue = value;
					unset = false;
				}
				double newValue = oldValue + alpha * (value - oldValue);
				oldValue = newValue;
			}

			double get() {
				return oldValue;
			}
		};

		struct Status {
			Status() :reset(true), lost(0) {}
			ExpMovingAverage k, b;
			bool reset;
			int lost;
		};

		Status laneR, laneL;

	public:
		laneDetection(double scaleFactor, Mat searchIMG)
		{
			if (!laneCascadeClassifier.load("haarcascade_cars3.xml")) { printf("--(!)Error loading - haarcascade_cars3.xml does not exist--(!) \n"); };
			laneScaleFactor = scaleFactor;

			src = searchIMG;
			halfWidth = (searchIMG.cols + 1) / 2;
			halfHeight = (searchIMG.rows + 1) / 2;
			startX = halfWidth - (halfWidth / 2);

			gaussianBlur_size = cv::Size(5, 5);
			gaussianBlur_sigmaX = 0;
			gaussianBlur_sigmaY = 0;
			gaussianBlur_borderType = cv::BorderTypes::BORDER_ISOLATED;

			canny_L2gradient = false;
			canny_threshold1 = 50;
			canny_threshold2 = 200;
			canny_aperturesize = 3;

			houghLinesP_threshold = 50;
			houghLinesP_minLineLength = 50;
			houghLinesP_maxLineGap = 100;
			houghLinesP_rho = 1;
			houghLinesP_theta = 1 * (CV_PI / 180);

			lineMinRejectDegree = 10;
			lineMaxRejectDegree = 89;

			SCAN_STEP = 5;			  // in pixels
			BW_TRESHOLD = 250;		  // edge response strength to recognize for 'WHITE'
			BORDERX = 10;			  // px, skip this much from left & right borders
			MAX_RESPONSE_DIST = 5;	  // px
		}

		Mat dstCanny;

		std::vector<cv::Rect> testLanes(bool equalizeHistogram, Mat searchIMG)
		{
			//std::vector<shared_ptr<cv::Rect>> laneArray;
			std::vector<cv::Rect> laneArray;

			double scale = 1.0;
			int minNeighbors = 2;

			Mat halfFrame(round((searchIMG.rows + 1) / 2), round((searchIMG.cols + 1) / 2), searchIMG.type());
			cv::pyrDown(searchIMG, halfFrame);

			Mat smallImg(round((searchIMG.rows + 1) / scale), round((searchIMG.cols + 1) / scale), searchIMG.type());

			if (equalizeHistogram)
			{
				cv::equalizeHist(searchIMG, searchIMG);
				cv::equalizeHist(smallImg, smallImg);
			}

			laneCascadeClassifier.detectMultiScale(searchIMG, laneArray, laneScaleFactor, minNeighbors, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

			cv::Rect detectedLane;
			int biggestSize = 0; int biggestIndex = -1;
			for (int indx = 0; indx < laneArray.size(); indx++)
			{
				detectedLane = laneArray[indx];
				Point detectedLaneCenter(round(detectedLane.x + detectedLane.width*0.5), round(detectedLane.y + detectedLane.height*0.5));
				int size = detectedLane.width*detectedLane.height;

				if (size > biggestSize)
				{
					biggestSize = size;
					biggestIndex = indx;
				}

				//rectangle(searchIMG, detectedLane, Scalar(255, 0, 255), 2, 8, 0);
				//ellipse(searchIMG, detectedLaneCenter, Size(detectedLane.width*0.25, detectedLane.height*0.25), 0, 0, 360, Scalar(255, 0, 255), 2, 8, 0);
			}

			if (biggestIndex != -1)
			{
				detectedLane = laneArray[biggestIndex];
				Point  Point1(round(detectedLane.x + detectedLane.width*0.25), round(detectedLane.y + detectedLane.height*0.25));
				Point  Point2(round(Point1.x + detectedLane.width*0.50), round(Point1.y + detectedLane.height*0.50));
				rectangle(searchIMG, Point1, Point2, Scalar(255, 255, 255), 2, 8, 0);
			}

			return laneArray;
		}

		/*std::vector<cv::Vec4i>*/ Mat getLanes(bool equalizeHistogram, int horizonLineIndex)
		{
			Mat srcBlurred;
			//Rect region_of_interest = Rect(0, halfHeight - 0, src.cols - 1, halfHeight - 1);
			int horizonLine = src.rows - horizonLineIndex;
			Rect region_of_interest = Rect(0, horizonLineIndex - 0, src.cols - 1, horizonLine - 1);

			Mat image_roi = src(region_of_interest);
			std::vector<cv::Vec4i> allLanes;
			int x1, x2, y1, y2;
			std::vector<Lane> left, right;

			GaussianBlur(image_roi, srcBlurred, gaussianBlur_size, gaussianBlur_sigmaX, gaussianBlur_sigmaY, gaussianBlur_borderType);
			Canny(srcBlurred, dstCanny, canny_threshold1, canny_threshold2, canny_aperturesize, canny_L2gradient);
			HoughLinesP(dstCanny, allLanes, houghLinesP_rho, houghLinesP_theta, houghLinesP_threshold, houghLinesP_minLineLength, houghLinesP_maxLineGap);
			Mat canny = dstCanny;

			int totalNumOfLanes = allLanes.size();

			cv::Mat colored_drawingImage, temp_frame;
			cv::cvtColor(src, colored_drawingImage, CV_GRAY2RGB);
			cv::cvtColor(src, temp_frame, CV_GRAY2RGB);

			for (size_t i = 1; i < totalNumOfLanes; i++)
			{
				x1 = allLanes[i][0];
				y1 = allLanes[i][1] + horizonLineIndex;
				x2 = allLanes[i][2];
				y2 = allLanes[i][3] + horizonLineIndex;

				Point point1(x1, y1);
				Point point2(x2, y2);

				int dx = x2 - x1;
				int dy = y2 - y1;
				float angle = atan2f(dy, dx) * 180 / CV_PI;

				// assign lane's side based by its midpoint position 
				int midx = (x1 + x2) / 2;

				if (((abs(angle) > lineMinRejectDegree)) && (abs(angle) < lineMaxRejectDegree))
				{
					dx = (dx == 0) ? 1 : dx; // prevent DIV/0!  
					float m = dy / float(dx);
					float b = y1 - m*x1;

					if (midx < halfWidth)
					{
						if (angle < 0)
						{
							left.push_back(Lane(point1, point2, angle, m, b));
						}

					}
					else
					{
						if (angle > 0)
						{
							right.push_back(Lane(point1, point2, angle, m, b));
						}
					}
					//BGR
					if ((y1 > (y2 + 50)) || (y1 < (y2 - 50)))
					{
						line(colored_drawingImage, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 255, 0), 1);
						//cout << "allLanes[" << i << "]" << " is O.K" << " x1:" << x1 << " y1:" << y1 << " x2:" << x2 << " y2:" << y2 << endl;
					}
					else
					{
						line(colored_drawingImage, Point(x1, y1), Point(x2, y2), cv::Scalar(255, 0, 0), 1);
						//cout << "allLanes[" << i << "]" << " (y1 <= (y2 + 50) && (y1 >= (y2 - 50))" << " x1:" << x1 << " y1:" << y1 << " x2:" << x2 << " y2:" << y2 << endl;
					}
				}
				else
				{
					line(colored_drawingImage, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 0, 255), 1);
					//cout << "allLanes[" << i << "]" << " angle is below 10" << " x1:" << x1 << " y1:" << y1 << " x2:" << x2 << " y2:" << y2 << endl;
				}
			}

			for (int i = 0; i<right.size(); i++) {
				line(temp_frame, right[i].p0, right[i].p1, cv::Scalar(0, 0, 255), 1);
				//cout << "right[" << i << "]" << " x1:" << right[i].p0.x << " y1:" << right[i].p0.x << " x2:" << right[i].p1.x << " y2:" << right[i].p1.y << " deg:" << right[i].angle << endl;
			}

			//cout << endl;

			for (int i = 0; i<left.size(); i++) {
				line(temp_frame, left[i].p0, left[i].p1, cv::Scalar(255, 0, 0), 1);
				//cout << "left[" << i << "]" << " x1:" << left[i].p0.x << " y1:" << left[i].p0.x << " x2:" << left[i].p1.x << " y2:" << left[i].p1.y << " deg:" << left[i].angle << endl;
			}

			if (left.size()>0)
			{
				processSide(left, dstCanny, false);
				// show computed lanes
				int xl1 = temp_frame.cols * 0;
				int xl2 = temp_frame.cols * 0.45f;
				double ml = laneL.k.get();
				double bl = laneL.b.get();
				Point pointL1(xl1, ml*xl1 + bl);
				Point pointL2(xl2, ml*xl2 + bl);
				line(temp_frame, pointL1, pointL2, cv::Scalar(0, 255, 0), 3);
			}

			if (right.size()>0)
			{
				processSide(right, dstCanny, true);
				// show computed lanes
				int xr1 = temp_frame.cols * 0.55f;
				int xr2 = temp_frame.cols - 1;
				double mr = laneR.k.get();
				double br = laneR.b.get();
				Point pointR1(xr1, mr*xr1 + br);
				Point pointR2(xr2, mr*xr2 + br);
				line(temp_frame, pointR1, pointR2, cv::Scalar(0, 255, 0), 3);
			}

			return temp_frame;
			//return allLanes;
			//src.SetROI(new CvRect(0, halfHeight - 0, src.Width - 1, src.Height - 1));
		}

		CvPoint2D32f sub(CvPoint2D32f b, CvPoint2D32f a) { return cvPoint2D32f(b.x - a.x, b.y - a.y); }
		CvPoint2D32f mul(CvPoint2D32f b, CvPoint2D32f a) { return cvPoint2D32f(b.x*a.x, b.y*a.y); }
		CvPoint2D32f add(CvPoint2D32f b, CvPoint2D32f a) { return cvPoint2D32f(b.x + a.x, b.y + a.y); }
		CvPoint2D32f mul(CvPoint2D32f b, float t) { return cvPoint2D32f(b.x*t, b.y*t); }
		float dot(CvPoint2D32f a, CvPoint2D32f b) { return (b.x*a.x + b.y*a.y); }
		float dist(CvPoint2D32f v) { return sqrtf(v.x*v.x + v.y*v.y); }

		CvPoint2D32f point_on_segment(CvPoint2D32f line0, CvPoint2D32f line1, CvPoint2D32f pt) {
			CvPoint2D32f v = sub(pt, line0);
			CvPoint2D32f dir = sub(line1, line0);
			float len = dist(dir);
			float inv = 1.0f / (len + 1e-6f);
			dir.x *= inv;
			dir.y *= inv;

			float t = dot(dir, v);
			if (t >= len) return line1;
			else if (t <= 0) return line0;

			return add(line0, mul(dir, t));
		}

		float dist2line(CvPoint2D32f line0, CvPoint2D32f line1, CvPoint2D32f pt) {
			return dist(sub(point_on_segment(line0, line1, pt), pt));
		}

		void FindResponses(Mat img, int startX, int endX, int y, std::vector<int>& list)
		{
			// scans for single response: /^\_

			const int row = y * img.cols * img.channels();
			unsigned char* ptr = static_cast<unsigned char*>(img.ptr());

			int step = (endX < startX) ? -1 : 1;
			int range = (endX > startX) ? endX - startX + 1 : startX - endX + 1;

			for (int x = startX; range>0; x += step, range--)
			{
				if (ptr[row + x] <= BW_TRESHOLD) continue; // skip black: loop until white pixels show up

														   // first response found
				int idx = x + step;

				// skip same response(white) pixels
				while (range > 0 && ptr[row + idx] > BW_TRESHOLD) {
					idx += step;
					range--;
				}

				// reached black again
				if (ptr[row + idx] <= BW_TRESHOLD) {
					list.push_back(x);
				}

				x = idx; // begin from new pos
			}
		}

		void processSide(std::vector<Lane> lanes, Mat edges, bool right)
		{

			Status* side = right ? &laneR : &laneL;

			// response search
			int w = edges.cols;
			int h = edges.rows;
			const int BEGINY = 0;
			const int ENDY = h - 1;
			const int ENDX = right ? (w - BORDERX) : BORDERX;
			int midx = w / 2;
			int midy = h / 2;
			unsigned char* ptr = static_cast<unsigned char*>(edges.ptr());

			// show responses
			int* votes = new int[lanes.size()];
			for (int i = 0; i<lanes.size(); i++) votes[i++] = 0;

			for (int y = ENDY; y >= BEGINY; y -= SCAN_STEP) {
				std::vector<int> rsp;
				FindResponses(edges, midx, ENDX, y, rsp);

				if (rsp.size() > 0) {
					int response_x = rsp[0]; // use first reponse (closest to screen center)

					float dmin = 9999999;
					float xmin = 9999999;
					int match = -1;
					for (int j = 0; j<lanes.size(); j++) {
						// compute response point distance to current line
						float d = dist2line(
							cvPoint2D32f(lanes[j].p0.x, lanes[j].p0.y),
							cvPoint2D32f(lanes[j].p1.x, lanes[j].p1.y),
							cvPoint2D32f(response_x, y));

						// point on line at current y line
						int xline = (y - lanes[j].b) / lanes[j].k;
						int dist_mid = abs(midx - xline); // distance to midpoint

														  // pick the best closest match to line & to screen center
						if (match == -1 || (d <= dmin && dist_mid < xmin)) {
							dmin = d;
							match = j;
							xmin = dist_mid;
							break;
						}
					}

					// vote for each line
					if (match != -1) {
						votes[match] += 1;
						lanes[match].votes += 1;
					}
				}
			}

			int bestMatch = -1;
			int mini = 9999999;
			for (int i = 0; i<lanes.size(); i++) {
				int xline = (midy - lanes[i].b) / lanes[i].k;
				int dist = abs(midx - xline); // distance to midpoint

				if (bestMatch == -1 || (votes[i] > votes[bestMatch] && dist < mini)) {
					bestMatch = i;
					mini = dist;
				}
			}

			if (bestMatch != -1) {
				Lane* best = &lanes[bestMatch];
				float k_diff = fabs(best->k - side->k.get());
				float b_diff = fabs(best->b - side->b.get());

				bool update_ok = (k_diff <= K_VARY_FACTOR && b_diff <= B_VARY_FACTOR) || side->reset;

				printf("side: %s, k vary: %.4f, b vary: %.4f, lost: %s\n",
					(right ? "RIGHT" : "LEFT"), k_diff, b_diff, (update_ok ? "no" : "yes"));

				if (update_ok) {
					// update is in valid bounds
					side->k.add(best->k);
					side->b.add(best->b);
					side->reset = false;
					side->lost = 0;
				}
				else {
					// can't update, lanes flicker periodically, start counter for partial reset!
					side->lost++;
					if (side->lost >= MAX_LOST_FRAMES && !side->reset) {
						side->reset = true;
					}
				}

			}
			else {
				printf("no lanes detected - lane tracking lost! counter increased\n");
				side->lost++;
				if (side->lost >= MAX_LOST_FRAMES && !side->reset) {
					// do full reset when lost for more than N frames
					side->reset = true;
					side->k.clear();
					side->b.clear();
				}
			}

			delete[] votes;
		}

		~laneDetection()
		{

		}

		void clear()
		{
		}
	};
#endif

	class MyObject
	{
	private:
		int avgDisparity = 0;
		int highestDisparity = 0;

	public:
		double PI = 3.141592653589793238463;

		int labelNum = -1;
		int mergedLabelNum = -1;
		int mergedLabelNumInclined = -1;
		bool isLeftSide = false;
		bool isLeftLeaning = false;
		bool isMainFrame = false;
		bool isProcessed = false;
		bool isSideBuilding = false;
		bool isReverseSide = false;
		bool isStraightLine = false;

		int xLow_Index = 0;//Leftline of the udisp & densDisp object
		int xHigh_Index = 0;//Rightline of the udisp & densDisp object
		int ylowDisp_Index = 128;//Upperline of the udisp object
		int yhighDisp_Index = 0; //Baseline of the udisp object

		float yhighDisp_Index_real = 0.0;
		float ylowDisp_Index_real = 0.0;

		int ylowDisp_Vmin_Index = 0; //Upperline of the densDisp object
		int yhighDisp_Vmax_Index = 128;//Baseline of the densDisp object
		int yhighDisp_Vmin_Index = 0; //Optional Upperline of the densDisp object
		int ylowDisp_Vmax_Index = 128;//Optional Baseline of the densDisp object

		int xLow_Index_lowDisp = 0;//keep the beginnig disparities to check if it is reverse Side
		int xHigh_Index_lowDisp = 0;//keep the beginnig disparities to check if it is reverse Side
		int xLow_Index_highDisp = 0;//keep the beginnig disparities to check if it is reverse Side
		int xHigh_Index_highDisp = 0;//keep the beginnig disparities to check if it is reverse Side

		bool isMerged = false;
		bool isMergedInclined = false;
		bool isDeleted = false;

		int quarterNum = 0;
		float realDistanceTiltCorrected = 0, directDistance = 0, depthAVG = 0;
		float Xw = 0, Yw = 0, Zw = 0, Xs = 0, Ys = 0, Zs = 0;
		float depth = 0;
		float objHeight = 0, objWidth = 0, objSize = 0, objRelativeVelocity = 0;
		int maskV = 0;
		int mainFrame_maskV = 0;
		int maskH = 0;
		float inclinedObjAnglePrecision = 0;

		float udisp_slope = -1;
		float ddisp_slope = -1;
		float ddisp_slope_b = -1;
		float udisp_slope_b = -1;

		float udisp_radian = -1;
		float ddisp_radian = -1;

		float udisp_degree = -1;
		float ddisp_degree = -1;

		cv::Point dDisp_lineLeftP1;
		cv::Point dDisp_lineLeftP2;
		cv::Point dDisp_lineRightP1;
		cv::Point dDisp_lineRightP2;
		cv::Point dDisp_lineUpP1;
		cv::Point dDisp_lineUpP2;
		cv::Point dDisp_lineBottomP1;
		cv::Point dDisp_lineBottomP2;

		cv::Point uDisp_lineLeftP1;
		cv::Point uDisp_lineLeftP2;
		cv::Point uDisp_lineRightP1;
		cv::Point uDisp_lineRightP2;
		cv::Point uDisp_lineUpP1;
		cv::Point uDisp_lineUpP2;
		cv::Point uDisp_lineBottomP1;
		cv::Point uDisp_lineBottomP2;

		MyObject()
		{
			//cout << "Init MyObject" << endl;
		}

		~MyObject()
		{
			//cout << "Deleting MyObject" << endl;
		}

		void set_dDispObjLine()
		{
			if ((xHigh_Index - xLow_Index) > 0)
			{
				if (isLeftLeaning)
				{
					ddisp_slope = (float(yhighDisp_Vmax_Index) - ylowDisp_Vmax_Index) / (xLow_Index - xHigh_Index);
					ddisp_slope_b = ylowDisp_Vmax_Index - ddisp_slope*xHigh_Index;
				}
				else
				{
					ddisp_slope = (float(ylowDisp_Vmax_Index) - yhighDisp_Vmax_Index) / (xLow_Index - xHigh_Index);
					ddisp_slope_b = ylowDisp_Vmax_Index - ddisp_slope*xLow_Index;
				}
			}
			else
				ddisp_slope = 0;
		}

		void setDensDispObject(bool isUpLineFoundByMaxD, cv::Mat &Global_Current_DenseDisparity_Image32F)
		{
			if (isDeleted) return;

			highestDisparity = xHigh_Index;

			float m_reverse, b;

			if (ddisp_slope == -1) set_dDispObjLine();
			m_reverse = -ddisp_slope;

			if (isUpLineFoundByMaxD)
			{
				if (isLeftLeaning)
				{
					b = yhighDisp_Vmin_Index - m_reverse*xLow_Index;
					ylowDisp_Vmin_Index = m_reverse*xHigh_Index + b;
				}
				else
				{
					b = yhighDisp_Vmin_Index - m_reverse*xHigh_Index;
					ylowDisp_Vmin_Index = m_reverse*xLow_Index + b;
				}
			}
			else
			{
				if (isLeftLeaning)
				{
					b = ylowDisp_Vmin_Index - m_reverse*xHigh_Index;
					yhighDisp_Vmin_Index = m_reverse*xLow_Index + b;
				}
				else
				{
					b = ylowDisp_Vmin_Index - m_reverse*xLow_Index;
					yhighDisp_Vmin_Index = m_reverse*xHigh_Index + b;
				}
			}

			if (yhighDisp_Vmin_Index < 0) yhighDisp_Vmin_Index = 0;

			if (isLeftLeaning)
			{
				dDisp_lineLeftP1.x = xLow_Index;
				dDisp_lineLeftP1.y = yhighDisp_Vmax_Index;
				dDisp_lineLeftP2.x = xLow_Index;
				dDisp_lineLeftP2.y = yhighDisp_Vmin_Index;

				//int h = yhighDisp_Index;
				//int l = ylowDisp_Index;

				yhighDisp_Index_real = *Global_Current_DenseDisparity_Image32F.ptr<const float>(yhighDisp_Vmin_Index, xLow_Index);
				ylowDisp_Index_real = *Global_Current_DenseDisparity_Image32F.ptr<const float>(ylowDisp_Vmin_Index, xHigh_Index);

				dDisp_lineRightP1.x = xHigh_Index;
				dDisp_lineRightP1.y = ylowDisp_Vmax_Index;
				dDisp_lineRightP2.x = xHigh_Index;
				dDisp_lineRightP2.y = ylowDisp_Vmin_Index;

				dDisp_lineUpP1.x = xLow_Index;
				dDisp_lineUpP1.y = yhighDisp_Vmin_Index;
				dDisp_lineUpP2.x = xHigh_Index;
				dDisp_lineUpP2.y = ylowDisp_Vmin_Index;

				dDisp_lineBottomP1.x = xLow_Index;
				dDisp_lineBottomP1.y = yhighDisp_Vmax_Index;
				dDisp_lineBottomP2.x = xHigh_Index;
				dDisp_lineBottomP2.y = ylowDisp_Vmax_Index;
			}
			else
			{
				dDisp_lineLeftP1.x = xLow_Index;
				dDisp_lineLeftP1.y = ylowDisp_Vmax_Index;
				dDisp_lineLeftP2.x = xLow_Index;
				dDisp_lineLeftP2.y = ylowDisp_Vmin_Index;

				yhighDisp_Index_real = *Global_Current_DenseDisparity_Image32F.ptr<const float>(yhighDisp_Vmin_Index, xHigh_Index);
				ylowDisp_Index_real = *Global_Current_DenseDisparity_Image32F.ptr<const float>(ylowDisp_Vmin_Index, xLow_Index);

				dDisp_lineRightP1.x = xHigh_Index;
				dDisp_lineRightP1.y = yhighDisp_Vmax_Index;
				dDisp_lineRightP2.x = xHigh_Index;
				dDisp_lineRightP2.y = yhighDisp_Vmin_Index;

				dDisp_lineUpP1.x = xHigh_Index;
				dDisp_lineUpP1.y = yhighDisp_Vmin_Index;
				dDisp_lineUpP2.x = xLow_Index;
				dDisp_lineUpP2.y = ylowDisp_Vmin_Index;

				dDisp_lineBottomP1.x = xHigh_Index;
				dDisp_lineBottomP1.y = yhighDisp_Vmax_Index;
				dDisp_lineBottomP2.x = xLow_Index;
				dDisp_lineBottomP2.y = ylowDisp_Vmax_Index;
			}

			if ((xHigh_Index - xLow_Index) > 0)
			{
				//ddisp_slope = ((float)dDisp_lineLeftP1.y - dDisp_lineRightP2.y) / (dDisp_lineRightP2.x - dDisp_lineLeftP1.x);
				ddisp_radian = std::atan(ddisp_slope);
				ddisp_degree = ddisp_radian*(180 / PI);
			}

			//float IDuvf = Global_Current_DenseDisparity_Image32F.ptr<const float>(u, v);
			//Global_pa



			StereoImageBuffer::ImageToWorldCoordinates(*this);
		}

		void setUdispObject(uDisparityParams &uparam)
		{
			if (isDeleted) return;

			unsigned int objMiddleX = round((xLow_Index + xHigh_Index) / 2.0);

			if (yhighDisp_Index == ylowDisp_Index)
			{
				isStraightLine = true;
			}
			else
			{
				isStraightLine = false;
			}


			if (objMiddleX < uparam.middleCol)
			{
				isLeftSide = true;

				if (objMiddleX >= uparam.mainFrameLeftCol)
				{
					isMainFrame = true;
				}
				else
				{
					isMainFrame = false;
				}

				//if (!(xLow_Index_highDisp < yhighDisp_Index))
				if ((xLow_Index_highDisp > ylowDisp_Index) || isStraightLine)
				{
					isLeftLeaning = true;
					isReverseSide = false;
				}
				else
				{
					isReverseSide = true;
					isLeftLeaning = false;
				}
			}
			else
			{
				isLeftSide = false;

				if (objMiddleX <= uparam.mainFrameRightCol)
				{
					isMainFrame = true;
				}
				else
				{
					isMainFrame = false;
				}

				//if (xLow_Index_lowDisp > ylowDisp_Index)
				if (xLow_Index_lowDisp > ylowDisp_Index)
				{
					isReverseSide = true;
					isLeftLeaning = true;
				}
				else
				{
					isReverseSide = false;
					isLeftLeaning = false;
				}
			}

			switch (quarterNum) {
			case 4:
			{
				maskH = uparam.maxD_MaskH;
				maskV = uparam.maxD_MaskV;
				mainFrame_maskV = uparam.mainFrame_maxD_MaskV;
				inclinedObjAnglePrecision = uparam.maxD_InclinedObject_AnglePrecision;
				break;
			}
			case 3:
			{
				maskH = uparam.threequarterD_MaskH;
				maskV = uparam.threequarterD_MaskV;
				mainFrame_maskV = uparam.mainFrame_threequarterD_MaskV;
				inclinedObjAnglePrecision = uparam.threequarterD_InclinedObject_AnglePrecision;
				break;
			}
			case 2:
			{
				maskH = uparam.halfD_MaskH;
				maskV = uparam.halfD_MaskV;
				mainFrame_maskV = uparam.mainFrame_halfD_MaskV;
				inclinedObjAnglePrecision = uparam.halfD_InclinedObject_AnglePrecision;
				break;
			}
			case 1:
			{
				maskH = uparam.quarterD_MaskH;
				maskV = uparam.quarterD_MaskV;
				mainFrame_maskV = uparam.mainFrame_quarterD_MaskV;
				inclinedObjAnglePrecision = uparam.quarterD_InclinedObject_AnglePrecision;
				break;
			}

			case 0:
			{
				maskH = uparam.zeroD_MaskH;
				maskV = uparam.zeroD_MaskV;
				mainFrame_maskV = uparam.mainFrame_zeroD_MaskV;
				inclinedObjAnglePrecision = uparam.zeroD_InclinedObject_AnglePrecision;
				break;
			}
			default: break;
			}


			if (isLeftLeaning)
			{
				uDisp_lineLeftP1.x = xLow_Index;
				uDisp_lineLeftP1.y = ylowDisp_Index;
				uDisp_lineLeftP2.x = xLow_Index;
				uDisp_lineLeftP2.y = yhighDisp_Index;

				uDisp_lineRightP1.x = xHigh_Index;
				uDisp_lineRightP1.y = ylowDisp_Index;
				uDisp_lineRightP2.x = xHigh_Index;
				uDisp_lineRightP2.y = yhighDisp_Index;
			}
			else
			{
				uDisp_lineLeftP1.x = xLow_Index;
				uDisp_lineLeftP1.y = yhighDisp_Index;
				uDisp_lineLeftP2.x = xLow_Index;
				uDisp_lineLeftP2.y = ylowDisp_Index;

				uDisp_lineRightP1.x = xHigh_Index;
				uDisp_lineRightP1.y = yhighDisp_Index;
				uDisp_lineRightP2.x = xHigh_Index;
				uDisp_lineRightP2.y = ylowDisp_Index;
			}

			uDisp_lineUpP1.x = xLow_Index;
			uDisp_lineUpP1.y = ylowDisp_Index;
			uDisp_lineUpP2.x = xHigh_Index;
			uDisp_lineUpP2.y = ylowDisp_Index;

			uDisp_lineBottomP1.x = xLow_Index;
			uDisp_lineBottomP1.y = yhighDisp_Index;
			uDisp_lineBottomP2.x = xHigh_Index;
			uDisp_lineBottomP2.y = yhighDisp_Index;

			if ((xHigh_Index - xLow_Index) > 0)
			{
				udisp_slope = (float(uDisp_lineLeftP1.y) - uDisp_lineRightP2.y) / (uDisp_lineRightP2.x - uDisp_lineLeftP1.x);
				udisp_radian = std::atan(udisp_slope);
				udisp_degree = udisp_radian*(180 / PI);
				udisp_slope_b = float(uDisp_lineLeftP1.y) - udisp_slope*uDisp_lineRightP2.x;

				if (isStraightLine)
				{
					if (isLeftLeaning)
						udisp_slope = -0.0001;
					else
						udisp_slope = 0.0001;
				}
			}
			else
			{
				isDeleted = true;
			}
		}

		int getAvgDisparity() const
		{
			if (yhighDisp_Index == ylowDisp_Index || (yhighDisp_Index - ylowDisp_Index) == 1) return yhighDisp_Index;
			else if ((yhighDisp_Index - ylowDisp_Index) % 2 == 0) return ((yhighDisp_Index + ylowDisp_Index) / 2);
			else return ((yhighDisp_Index + ylowDisp_Index + 1) / 2);
		}

		int getDensdisp_Length() const
		{
			return abs(dDisp_lineLeftP1.x - dDisp_lineRightP1.x);
		}

		int getDensdisp_Height() const
		{
			return abs(dDisp_lineLeftP1.y - dDisp_lineLeftP2.y);
		}

		int getUdisp_Length() const
		{
			return abs(uDisp_lineLeftP1.x - uDisp_lineRightP1.x);
		}

		int getUdisp_Height() const
		{
			return abs(uDisp_lineLeftP1.y - uDisp_lineLeftP2.y);
		}

	};

#ifdef WITH_DOCK_DETECTION
	class MyLine
	{
	public:
		float x1, y1, x2, y2, realBottom_x, realBottom_y, realUp_x, realUp_y;
		float m;
		float b;
		float radian;
		float degree;
		float length;
		float wallRatio; //ratio of how similar to a rectangle
		bool signMinus;
		int index;
		float realwidthDistance;
		float realHeight;
		float d;

		MyLine()
		{
			x1 = 0;
			y1 = 0;
			x2 = 0;
			y2 = 0;
			m = 0;
			b = 0;
			d = 0;
			radian = 0;
			degree = 0;
			length = 0;
			wallRatio = -1;
			signMinus = false;
			index = 0;
		}

		MyLine(cv::Vec4i line)
		{
			x1 = line[0];
			y1 = line[1];
			x2 = line[2];
			y2 = line[3];
			this->makeline();
		}

		void drawLine(MyLine myline, Mat& img)
		{
			int r = rand() % 255;
			int g = rand() % 255;
			int b = rand() % 255;

			cv::line(img, Point(myline.x1, myline.y1), Point(myline.x2, myline.y2), Scalar(r, g, b), 1, cv::LineTypes::LINE_4);

			std::ostringstream text;
			text << myline.index << ":" << myline.degree << " x1 " << myline.x1 << "," << myline.y1;
			cv::putText(img, text.str(), Point(myline.x1 + 50, myline.y1), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(r, g, b), 1, CV_AA);
			text.str("");
			text.clear();

			text << myline.index << ":" << myline.degree << " x2 " << myline.x2 << "," << myline.y2;
			cv::putText(img, text.str(), Point(myline.x2 - 200, myline.y2), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(r, g, b), 1, CV_AA);
			text.str("");
			text.clear();
		}

		void drawLine(Vec4i line, Mat& img)
		{
			float x1 = line[0];
			float y1 = line[1];
			float x2 = line[2];
			float y2 = line[3];

			int r = rand() % 255;
			int g = rand() % 255;
			int b = rand() % 255;

			cv::line(img, Point(x1, y1), Point(x2, y2), Scalar(r, g, b), 1, cv::LineTypes::LINE_4);

			float radian = std::atan2((y2 - y1), (x2 - x1));
			float radian2 = std::atan((y2 - y1) / (x2 - x1));
			float degree = radian2*(180 / CV_PI);

			float length = sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));

			std::ostringstream text;
			text << 0 << ":" << degree << " x1 " << x1 << "," << y1;
			cv::putText(img, text.str(), Point(x1 + 50, y1), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(r, g, b), 1, CV_AA);
			text.str("");
			text.clear();

			text << 0 << ":" << degree << " x2 " << x2 << "," << y2;
			cv::putText(img, text.str(), Point(x2 - 200, y2), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(r, g, b), 1, CV_AA);
			text.str("");
			text.clear();
		}

		void makeline()
		{
			if (y1 > y2)//y1 is the top then flip points (this process does not change anything regarding to the line function)
			{
				float tempX1 = x1;
				float tempY1 = y1;

				x1 = x2;
				x2 = tempX1;
				y1 = y2;
				y2 = tempY1;
			}

			m = ((y2 - y1) / (x2 - x1));//inf if it is 90 degree (-inf if it is -90 degree)
			b = y1 - m*x1;
			radian = std::atan(m);//no need to tan2, we only need (-90)-(+90) degree interval
			degree = radian*(180 / CV_PI);
			length = sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));
			signMinus = signbit(radian);
		}

	};

	class MyDockObject
	{
	public:
		float d, u, v;
		float Xw = -1, Yw = -1, Zw = -1, Xs = -1, Ys = -1, Zs = -1;
		float distanceW = -1, distanceS = -1;

		MyDockObject(float disparity, float xCoordinate, float yCoordinate)
		{
			d = disparity;
			u = xCoordinate;
			v = yCoordinate;
		}

	};

#endif

	//directories
	//const std::string img_root_dir = "E:\\Depo\\KUDiary\\Important Doc\\3D Detection\\stereo Images\\daimlerUrbanSeg\\daimlerUrbanSeg";
	//const std::string img_root_dir = "E:\\KUDiary\\Important Doc\\3D Detection\\stereo Images\\daimlerUrbanSeg\\daimlerUrbanSeg";

	std::string img_root_dir;// = "E:\\KUDiary\\Important Doc\\3D Detection\\stereo Images\\Bumblebee";
							 //const std::string img_root_dir = "E:\\Depo\\KUDiary\\Important Doc\\3D Detection\\stereo Images\\Bumblebee";
	std::string img_GT_root_dir = "E:\\Depo\\KUDiary\\Important Doc\\GPU\\Disparity Tests\\NTSD-200";
	std::string Global_LeftIMG_Dir, Global_RightIMG_Dir, Global_LeftIMG_GTDir, Global_RightIMG_GTDir, Global_LeftDisparity_GTDir, Global_RightDisparity_GTDir;

	const double PI = 3.141592653589793238463;

	int Global_Buffer_Size;

	int Global_currentIMGNum;
	int Global_minIMGIndex;
	int Global_maxIMGIndex;
	int Global_imgIncrement;

	int Global_imgTotalRows;
	int Global_imgTotalCols;
	int Global_imgMiddleCol;

	bool Global_CropRectifiedIMG = false;
	unsigned int Global_startU = 24;
	unsigned int Global_endU = 1000;
	unsigned int Global_startV = 40;
	unsigned int Global_endV = 400;//370;

	vDisparityParams *Global_vDisparityParams;
	vDisparityGlobals *Global_vDisparityGlobals;
	uDisparityParams *Global_uDisparityParams;
	uDisparityGlobals *Global_uDisparityGlobals;
	dDisparityParams *Global_dDisparityParams;
	dDisparityGlobals *Global_dDisparityGlobals;


#ifdef WITH_OCCUPANCY_GRID
	occupancyGridParams *Global_occupancyGridParams;
	occupancyGridGlobals *Global_occupancyGridGlobals;
#endif

#ifdef WITH_BUMBLEBEE
	Bumblebee *bumblebeeObject;
#endif

	int Global_maxNumOfObjectAllowed; //so maxLabel == 255
	std::vector<MyObject, NAlloc<MyObject>> Global_labeledObjectList;
	std::vector<MyObject, NAlloc<MyObject>> Global_mergedObjectList1;
	std::vector<MyObject, NAlloc<MyObject>> Global_mergedObjectList2;

	//delete later just for visualiztion
	class myDraw
	{
	public:
		int imgNum;
		cv::Point2i p1;
		cv::Point2i p2;
		string text;
		float distance;
		float Pou;
		cv::Point2i closes_Point;

		myDraw(int imNum, string txt, float dist, float P, cv::Point2i np1, cv::Point2i np2, cv::Point2i cPoint)
		{
			imgNum = imNum;
			text = txt;
			p1 = np1;
			p2 = np2;
			closes_Point = cPoint;
			distance = dist;
			Pou = P;
		}
	};

	std::vector<myDraw *> drawings;
	//delete later just for visualiztion

	//std::vector<MyObject> Global_mergedObjectList3;

	mutable unsigned int Global_currentNumOfObjects;

	int Global_numDisparities;
	int Global_currentIMGIndex;
	bool Global_useGPU;
	bool Global_useBumblebee;
	bool Global_bumblebee_widemode;

	vector<shared_ptr<cv::Mat>> Global_IMG_Left_Image_VectorOfSharedPtr;
	vector<shared_ptr<cv::Mat>> Global_IMG_Right_Image_VectorOfSharedPtr;

	cv::Mat Global_Current_Left_Image;
	cv::Mat Global_Current_Right_Image;
	cv::Mat Global_Current_DenseDisparity_Image;//use to create u-v disparity
	cv::Mat Global_Current_DenseDisparity_Image32F;//use to find precise disparities
	cv::Mat mapPath_img;

	float Global_imgScale = 1.0;
	bool Global_equalizeHist = false;
	bool Global_rectify = false;
	bool Global_useRealTime = false;

#ifdef WITH_GPU
	cv::Mat *Global_IMG_Left_Image_Buffer_GPU;
	cv::Mat *Global_IMG_Right_Image_Buffer_GPU;
#endif

#ifdef WITH_SLAM //manual angles -->delete later
	double * manualAngles;
	double * manualX;
	double * manualZ;
#endif

public:
	//static stereoCamParams stereoParams;

	StereoImageBuffer(std::string seq, int minIMGIndex, int maxIMGIndex, int imgIncrement, bool useGPU, int numOfDisparity, float scale, bool use_Bumblebee, bool bumblebee_widemode)
	{
		Global_useRealTime = false;

		if (use_Bumblebee)
		{
			//img_root_dir = "E:\\KUDiary\\Important Doc\\3D Detection\\stereo Images\\Bumblebee";
			img_root_dir = "/home/vefak/sancaktepe";
		}
		else
		{
			img_root_dir = "E:\\KUDiary\\Important Doc\\3D Detection\\stereo Images\\daimlerUrbanSeg\\daimlerUrbanSeg";
		}

		Global_LeftIMG_Dir = img_root_dir + "\\" + seq;
		Global_RightIMG_Dir = img_root_dir + "\\" + seq;

		Global_minIMGIndex = minIMGIndex;
		Global_maxIMGIndex = maxIMGIndex;
		Global_imgIncrement = imgIncrement;
		Global_useGPU = useGPU;
		Global_currentIMGIndex = 0;
		Global_imgScale = scale;
		Global_useBumblebee = use_Bumblebee;
		Global_bumblebee_widemode = bumblebee_widemode;

		Global_numDisparities = numOfDisparity;
		Global_maxNumOfObjectAllowed = numeric_limits<unsigned char>::max(); //so maxLabel == 255

#ifdef WITH_SLAM //manual angles -->delete later
		manualAngles = new double[32]{ 0.0, -45.0, -67.5, -90.0, -112.5, -135.0, -157.5, -180, 157.5, 135.0, 112.5, 90.0, 67.5, 45, 22.5, 0.0, 0.0, -45.0, -67.5, -90.0, -112.5, -135.0, -157.5, -180, 157.5, 135.0, 112.5, 90.0, 67.5, 45, 22.5, 0.0 };
		//manualZ = new double[32]{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.2, -3.2, -3.2, -3.2, -3.2, -3.2, -3.2, -3.2, -3.2, -3.2,-3.2, -3.2, -3.2, -3.2, -3.2, -3.2 };
		manualZ = new double[32]{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,2.0, 2.0, 2.0, 2.0, 2.0, 2.0 };
		//manualZ = new double[32]{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0,-2.0, -2.0, -2.0, -2.0, -2.0, -2.0 };
		//manualZ = new double[32]{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.65, -1.65, -1.65, -1.65, -1.65, -1.65, -1.65, -1.65, -1.65, -1.65,-1.65, -1.65, -1.65, -1.65, -1.65, -1.65 };
		//manualZ = new double[32]{ -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.65, -1.65, -1.65, -1.65, -1.65, -1.65, -1.65, -1.65, -1.65, -1.65,-1.65, -1.65, -1.65, -1.65, -1.65, -1.65 };
		//manualZ = new double[32]{ 1.65, 1.65, 1.65, 1.65, 1.65, 1.65, 1.65, 1.65, 1.65, 1.65, 1.65, 1.65, 1.65, 1.65, 1.65, 1.65, -1.65, -1.65, -1.65, -1.65, -1.65, -1.65, -1.65, -1.65, -1.65, -1.65,-1.65, -1.65, -1.65, -1.65, -1.65, -1.65 };
		//manualZ = new double[32]{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		//manualZ = new double[32]{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.2, -3.2, -3.2, -3.2, -3.2, -3.2, -3.2, -3.2, -3.2, -3.2,-3.2, -3.2, -3.2, -3.2, -3.2, -3.2 };
		//manualX = new double[32]{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2,3.2, 3.2, 3.2, 3.2, 3.2, 3.2 };
		//manualX = new double[32]{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0,-2.0, -2.0, -2.0, -2.0, -2.0, -2.0};
		//manualX = new double[32]{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2 };
		//manualX = new double[32]{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		//manualX = new double[32]{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1 };
		//manualX = new double[32]{ -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1 };
		//manualX = new double[32]{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 };
		//manualX = new double[32]{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0 };
		//manualX = new double[32]{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.35, -2.35, -2.35, -2.35, -2.35, -2.35, -2.35, -2.35, -2.35, -2.35, -2.35, -2.35, -2.35, -2.35, -2.35, -2.35 };
		manualX = new double[32]{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.2, -3.2, -3.2, -3.2, -3.2, -3.2, -3.2, -3.2, -3.2, -3.2, -3.2, -3.2, -3.2, -3.2, -3.2, -3.2 };
		//manualZ = new double[32]{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7,-7, -7, -7, -7, -7, -7 };

#endif

		setBufferSize();
		const int bufferSize = getBufferSize();

		if (useGPU)
		{
#ifdef WITH_GPU
			Global_IMG_Left_Image_Buffer_GPU = new cv::Mat[bufferSize];
			Global_IMG_Right_Image_Buffer_GPU = new cv::Mat[bufferSize];
#else
			cout << "'#define WITH_GPU' is missing" << endl;
#endif
		}

		setDefaultParams();
		float ratio = 2.0;
		int rD = round(Global_dDisparityParams->totalRows / ratio);
		int cD = round(Global_dDisparityParams->totalCols / ratio);

		int rU = round(Global_uDisparityParams->totalRows / ratio);
		int cU = round(Global_uDisparityParams->totalCols / ratio);

		int rV = round(Global_vDisparityParams->totalRows / ratio);
		int cV = round(Global_vDisparityParams->totalCols / ratio);

		//cv::namedWindow("Window1", WINDOW_NORMAL);		
		//cv::resizeWindow("Window1", cU, rU*5);
		//cv::moveWindow("Window1", 1210, 10);

		//cv::namedWindow("Window2", WINDOW_NORMAL);
		//cv::resizeWindow("Window2", cU, rU);
		//cv::moveWindow("Window2", 10, 200);

		//cv::namedWindow("Window3", WINDOW_NORMAL);
		//cv::resizeWindow("Window3", cD, rD);
		//cv::moveWindow("Window3", 10, 300);

		//cv::namedWindow("Window4", WINDOW_NORMAL);
		//cv::resizeWindow("Window4", cV, rV);
		//cv::moveWindow("Window4", 910, 300);

		//cv::namedWindow("Window5", WINDOW_NORMAL);
		//cv::resizeWindow("Window5", cU, rU);
		//cv::moveWindow("Window5", 10, 950);

		cv::namedWindow("Window1", WINDOW_NORMAL);
		cv::resizeWindow("Window1", cU, rU);
		cv::moveWindow("Window1", 10, 10);

		//cv::namedWindow("Window2", WINDOW_NORMAL);
		//cv::resizeWindow("Window2", cU, rU);
		//cv::moveWindow("Window2", 110, 10);

		//cv::namedWindow("Window3", WINDOW_NORMAL);
		//cv::resizeWindow("Window3", cD, rD);
		//cv::moveWindow("Window3", 1210, 450);



		//cv::namedWindow("Window4", WINDOW_NORMAL);
		//cv::resizeWindow("Window4", cV, rV);
		//cv::moveWindow("Window4", 910, 300);

		//cv::namedWindow("Window5", WINDOW_NORMAL);
		//cv::resizeWindow("Window5", cU, rU);
		//cv::moveWindow("Window5", 10, 950);

#ifdef WITH_OCCUPANCY_GRID

		float width = Global_occupancyGridParams->XWorldLimit * 2;
		float height = Global_occupancyGridParams->ZWorldLimit;
		int occupancyTotalCols = round(width / Global_occupancyGridParams->XWorldResolution);
		int occupancyTotalRows = round(height / Global_occupancyGridParams->ZWorldResolution);
		int rC = round(occupancyTotalRows * ratio);
		int cC = round(occupancyTotalCols * ratio);

		//cv::namedWindow("Window6", WINDOW_NORMAL);
		//cv::resizeWindow("Window6", cC, rC);
		//cv::moveWindow("Window6", 1210, 10);

		//rC = round(occupancyTotalRows * ratio);
		//cC = round(occupancyTotalCols * ratio);

		//cv::namedWindow("Window7", WINDOW_NORMAL);
		//cv::resizeWindow("Window7", cC, rC);
		//cv::moveWindow("Window7", 10 + cC + 10, 10);

		//cv::namedWindow("Window1", WINDOW_NORMAL);
		//cv::resizeWindow("Window1", occupancyTotalCols, occupancyTotalRows);
		//cv::moveWindow("Window1", 1310, 10);

		cv::namedWindow("Window2", WINDOW_NORMAL);
		cv::resizeWindow("Window2", cC, rC);
		cv::moveWindow("Window2", 10, rU + 80);

		cv::namedWindow("Window3", WINDOW_NORMAL);
		cv::resizeWindow("Window3", occupancyTotalCols, occupancyTotalRows);
		cv::moveWindow("Window3", cC + 10, rU);

		cv::namedWindow("Window4", WINDOW_NORMAL);
		cv::resizeWindow("Window4", cD, rD);
		cv::moveWindow("Window4", cC + 10, occupancyTotalRows + 80);

		//cv::namedWindow("Window7", WINDOW_NORMAL);
		//cv::resizeWindow("Window7", Global_dDisparityParams->totalCols, Global_dDisparityParams->totalRows);
		//cv::moveWindow("Window7", 10, 100);
#endif

#ifdef WITH_LANE_DETECTION
		cv::namedWindow("Window6");
		cv::moveWindow("Window6", 910, 10);
		cv::namedWindow("Window7");
		cv::moveWindow("Window7", 1110, 410);
#endif
		//cv::namedWindow("Window8", WINDOW_NORMAL);
		//cv::resizeWindow("Window8", (cD*0.6), (rD*0.6));
		//cv::moveWindow("Window8", 1210, 650);
	}

	StereoImageBuffer(std::string seq, std::string condition, int minIMGIndex, int maxIMGIndex, int imgIncrement, bool useGPU, int numOfDisparity, float scale)
	{
		Global_useRealTime = false;

		Global_LeftIMG_Dir = img_root_dir + "\\" + seq;
		Global_RightIMG_Dir = img_root_dir + "\\" + seq;

		Global_LeftIMG_GTDir = img_GT_root_dir + "\\" + condition;
		Global_RightIMG_GTDir = img_GT_root_dir + "\\" + condition;

		Global_LeftDisparity_GTDir = img_GT_root_dir + "\\disparity_maps\\left";
		Global_RightDisparity_GTDir = img_GT_root_dir + "\\disparity_maps\\right";

		Global_minIMGIndex = minIMGIndex;
		Global_maxIMGIndex = maxIMGIndex;
		Global_imgIncrement = imgIncrement;
		Global_useGPU = useGPU;
		Global_currentIMGIndex = 0;

		Global_numDisparities = numOfDisparity;
		Global_maxNumOfObjectAllowed = numeric_limits<unsigned char>::max(); //so maxLabel == 255

		setBufferSize();
		const int bufferSize = getBufferSize();

		if (useGPU)
		{
#ifdef WITH_GPU
			Global_IMG_Left_Image_Buffer_GPU = new cv::Mat[bufferSize];
			Global_IMG_Right_Image_Buffer_GPU = new cv::Mat[bufferSize];
#else
			cout << "'#define WITH_GPU' is missing" << endl;
#endif
		}

		setDefaultParams();

		cv::namedWindow("Window1");
		cv::moveWindow("Window1", 50, 50);

		cv::namedWindow("Window2");
		cv::moveWindow("Window2", 50, 200);

		cv::namedWindow("Window3");
		cv::moveWindow("Window3", 50, 350);

		cv::namedWindow("Window4");
		cv::moveWindow("Window4", 1150, 350);

		cv::namedWindow("Window5");
		cv::moveWindow("Window5", 50, 750);

		cv::namedWindow("Window6");
		cv::moveWindow("Window6", 1200, 50);
	}

	~StereoImageBuffer()
	{
		//cout << "StereoImageBuffer Object is being deleted... Please wait" << endl;

		delete Global_vDisparityGlobals;
		Global_vDisparityGlobals = nullptr;

		delete Global_vDisparityParams;
		Global_vDisparityParams = nullptr;

		delete Global_uDisparityGlobals;
		Global_uDisparityGlobals = nullptr;

		delete Global_uDisparityParams;
		Global_uDisparityParams = nullptr;

		delete Global_dDisparityGlobals;
		Global_dDisparityGlobals = nullptr;

		delete Global_dDisparityParams;
		Global_dDisparityParams = nullptr;

#ifdef WITH_OCCUPANCY_GRID
		delete Global_occupancyGridParams;
		Global_occupancyGridParams = nullptr;

		delete Global_occupancyGridGlobals;
		Global_occupancyGridGlobals = nullptr;
#endif

#ifdef WITH_GPU
		if (Global_useGPU)
		{
			delete[] Global_IMG_Left_Image_Buffer_GPU;
			Global_IMG_Left_Image_Buffer_GPU = nullptr;

			delete[] Global_IMG_Right_Image_Buffer_GPU;
			Global_IMG_Right_Image_Buffer_GPU = nullptr;
		}
#endif
		//cout << "StereoImageBuffer Object is deleted" << endl;
	}

#pragma region Function Declarations
	//static void StereoImageBuffer::ImageToWorldCoordinates(StereoImageBuffer::MyObject& obj); - Oguz
	static void ImageToWorldCoordinates(StereoImageBuffer::MyObject& obj); 

	int getNumObjectsDeleted(vector<MyObject, NAlloc<MyObject>>& objects) const;
	void printObject(MyObject& obj) const;

	void setDefaultParams();
	void getImagePaths(bool reverse, int imgNum, std::string& fname_image_left, std::string& fname_image_right) const;
	void setImages(bool reverse, bool equalizeHist = false, bool rectify = false);
	void processImageDisparities(bool showImages);

	void setBufferSize();
	int getBufferSize() const;

	double getPI() const;
	ostringstream getCurrentImageNum() const;
	int getCurrentImageNumInt() const;
	void set_DenseDisparityMap();
	void set_UDisparityMap();
	void set_VDisparityMap();
	void set_UVDisparityMaps();
	void set_BestLine_From_VDisparity() const;
	void set_BestLine_From_VDisparity_Curve() const;
	void set_BestLine_From_VDisparity_Regression() const;
	void set_ImageInfinityPoint_And_VehicleRoadLine(int infinityPointX, int infinityPointY, int roadLineBottomX, int roadLineBottomY);//X=col, Y=row

	void set_LabelsOf_UDisparityLines_Using_LabelingAlg();//prepare and call the subset_LabelingAlg()
	void subset_LabelingAlg(int quarterNum, int horizontalMask, int verticalMask, int mainFrame_verticalMask, int current_u, int current_d, bool isPreviousMaskOccupied, int &out_labelNum, bool& out_onceOccupied, int& out_previousRightMostOccupiedCellIndex, std::vector<MyObject, NAlloc<MyObject>>& myGlobal_currentObjectList/*std::vector<MyObject>& myGlobal_currentObjectList*/) const;
	void merge_MaskVObjects_From_UDisparity_By_Labeling_Alg(std::vector<MyObject, NAlloc<MyObject>>& previousObjectList);
	void merge_InclinedObjects_From_UDisparity_By_Labeling_Alg(std::vector<MyObject, NAlloc<MyObject>>& previousObjectList);
	void updateMergingObjects(int new_xLow, int new_xHigh, int new_ylow, int new_yhigh, bool use_search_object_params, bool use_search_object_xlowD, int& currentLabelNum, MyObject& out_obj, MyObject& out_sobj, std::vector<MyObject, NAlloc<MyObject>>& out_newObjectList, bool reload) const;
	void set_SideBuildingsIfAny_And_Filter(std::vector<MyObject, NAlloc<MyObject>>& Global_mergedObjectList2);
	void createDenseDispObjects(std::vector<MyObject, NAlloc<MyObject>>& uDispObjectList);
	void get_Upperline_From_denseDisparity(MyObject& obj);
	Mat drawUDispObjects(Mat uDisp, std::vector<MyObject, NAlloc<MyObject>>& objects) const;
	cv::Mat drawUDispObjects3(cv::Mat uDisp, std::vector<MyObject, NAlloc<MyObject>>& objects, int startIndex, int endIndex) const;
	void addDrawPoint(int imgNum, string txt, int z, int x, int w, int h, Mat & occ, Mat & slam, float d);
	//cv::Mat drawUDispObjects2(cv::Mat uDisp, std::vector<MyObject, NAlloc<MyObject>>& objects, int numOfObjects) const;
	Mat drawObjects(cv::Mat drawingImage, std::vector<MyObject, NAlloc<MyObject>>& objects) const;
	bool checkIfObjectIsOnTheRoad(MyObject obj) const;
	int get_d_From_VDisparity_RoadProfile(float v) const;
	int get_v_From_VDisparity_RoadProfile(float d) const;

	float get_v_From_Calibration(float d, float Yw, int& out_v) const;
	float get_u_From_Calibration(float d, float Xw, int& out_u) const;
	float get_d_From_Calibration(float Zw, float Yw, int& out_d) const;

	static void ImageToWorldCoordinates(float u, float v, float d, float & out_Xw, float & out_Yw, float & out_Zw);

	float get_zw_From_Calibration(float d, float Yw) const;

	static void WorldToImageCoordinates(float Xw, float Yw, float Zw, float & out_u, float & out_v, float & out_d);

	void WorldToImageCoordinates(float Xw, float Yw, float Zw, int & out_u, int & out_v, int & out_d);

	void WorldToImagePlane(float Xw, float Yw, float Zw, float & out_Xs, float & out_Ys, float & out_Zs);

	static void ImageToImageCoordinates(float u, float v, float d, float & out_Xs, float & out_Ys, float & out_Zs);

	void cvPolyfit(cv::Mat & src_x, cv::Mat & src_y, cv::Mat & dst, int order) const;
	void testPolyFit() const;

#ifdef WITH_DOCK_DETECTION
	static void ImageToWorldCoordinates(MyDockObject &obj);
	void getMiddleRadian(MyLine& L1, MyLine& L2, float &difRadian, float &middleRadian) const;
	void getAngleBetweenlines(MyLine& L1, MyLine& L2, float &difRadian, float &middleRadian) const;
	void drawLine(float angle, float startX, float startY, float endY, Mat &draw) const;
	void drawLine(MyLine myline, Mat& img) const;
	void drawLine(Vec4i line, Mat& img) const;
	void equalizeLines(MyLine &line1, MyLine &line2) const;
	void getBestIndex(std::vector<MyLine>& lineArray, Mat &disparityIMG, float & bestScore, int & bestIndexL1, int & bestIndexL2, float rectangleRatio, float ratioThreshold, int middleY, int banIndexL1, int banIndexL2) const;
	MyLine merge2Lines(MyLine& L1, MyLine& L2) const;
	std::vector<MyLine> StereoImageBuffer::filterAndMakeLines(std::vector<cv::Vec4i> allLines, float minLength, float minDegree, Mat& imgTemp) const;
	int mergeLines(std::vector<cv::Vec4i> allLines, Mat& img, Mat& imgOrg, Mat& disparityIMG, std::vector<MyLine> &rectangle1, std::vector<MyLine> &rectangle2, Mat& occupancyMatrix, Mat& line_draws) const;
	void printHoughLines(std::vector<cv::Vec4i> allLines, Mat& img) const;
	void getBoundingBoxes(Mat& src, Mat& disp, Mat& SURF, Mat& occupancyMatrix, Mat& line_draws) const;
#endif

#ifdef WITH_GPU
	void setDenseDisparityMapCUDA();
#endif

#ifdef WITH_OCCUPANCY_GRID
	void set_occupancyUDisparityMap() const;
	void set_occupancyDDisparityMap() const;
	static void ImageToWorldCoordinates(StereoImageBuffer::uOccupancyCell& obj);
	float static my_round(float x, unsigned digits, bool recursive);
#endif

#ifdef WITH_BUMBLEBEE
	void setBumblebeeStereoParams(Bumblebee& obj);
#endif
#pragma endregion Function Declarations
};


class MyObject
{
public:
	float d, u, v;
	float Xw = -1, Yw = -1, Zw = -1, Xs = -1, Ys = -1, Zs = -1;
	float distanceW = -1, distanceS = -1;

	MyObject(float disparity, float xCoordinate, float yCoordinate)
	{
		d = disparity;
		u = xCoordinate;
		v = yCoordinate;
	}

};


struct ImageContainer
{
	FC2::Image tmp[2];
	FC2::Image unprocessed[2];
};





#endif
