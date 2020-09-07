#include "stereocore_RT.h"
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
		//OpenCVSGBMMode SGBM_Mode = OCV_SGBM;//5 ways default
		OpenCVSGBMMode SGBM_Mode = OCV_SGBM_3WAY;//5 ways default 
	} SGBM_Params;

	Bumblebee()
	{

	}

	Bumblebee(bool wideMode, resolutions res, int numDisp, int blockSize)
	{
		SGBM_Params.blockSize = blockSize;
		SGBM_Params.numDisp = numDisp;
		SGBM_Params.SGBM_P1 = SGBM_Params.blockSize * SGBM_Params.blockSize * 8;
		SGBM_Params.SGBM_P2 = SGBM_Params.blockSize * SGBM_Params.blockSize * 32;

		if (wideMode)
		{
            #ifdef WITH_BUMBLEBEE_REALTIME
				cameraStereoMode = FC2T::StereoCameraMode::TWO_CAMERA_WIDE;
				cameraStereoCalculationMode = TriclopsCameraConfiguration::TriCfg_2CAM_HORIZONTAL_WIDE;	
	        #endif

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
            #ifdef WITH_BUMBLEBEE_REALTIME
			cameraStereoMode = FC2T::StereoCameraMode::TWO_CAMERA_NARROW;
			cameraStereoCalculationMode = TriclopsCameraConfiguration::TriCfg_2CAM_HORIZONTAL_NARROW;
            #endif

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

	};



    	////Local Params
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
		float cY = 1.05; //1.61; //1.05;// 1.61;//1.0; // camera height above ground [meter]
		float cZ = 0.0;// 1.5;// 0.0; // camera offset Z [meter]
		float tilt = 0.0; // -0.0001;// 0.0;// -0.0001; //-0.095; //camera tilt angle [rad] (NOTE: should be adjusted by online tilt angle estimation)
		float cR = 0.0;//Pinhole center Row
		float cC = 0.0;//Pinhole center Col
	} stereoParams;


	bool setUpDefaults()
	{

	}
};