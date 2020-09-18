#include "Bumblebee_RT.h"



Mat convertTriclops2OpencvMat(TriclopsImage16& disparityImage);
Mat convertTriclops2OpencvMat(TriclopsImage& inputImage);
void ImageToWorldCoordinates(MyObject &obj, Bumblebee &bumbleObj);

void generatemonoStereoPair(TriclopsContext const &context,FC2::Image const &grabbedImage,	ImageContainer &imageCont, TriclopsMonoStereoPair &stereoPair);
void setImagePaths(int imgNum, std::string &fname_image_left, std::string &fname_image_right, std::string &fname_image_left_rectified, std::string &fname_image_right_rectified, std::string &fname_image_disparity, std::string &fname_openCV_image_disparity);
std::string getStr(int def);
std::string getStr(float def);
std::string getStr(double def);
void createDirectoryIf(string strPath);

// struct containing image needed for processing


enum IMAGE_SIDE
{
	RIGHT = 0, 
	LEFT
};


const std::string img_root_dir = "/home/vefak/Desktop";
std::string Global_LeftIMG_Dir, Global_RightIMG_Dir, Global_DisparityIMG_Dir, Global_OpenCVDisparityIMG_Dir;
std::string seq = "ofis_test";


int main()
{
	bool useOfflineCode = false;
	clock_t t_start, t_end;
	double diff_disparity;

	if (useOfflineCode)
	{
		getchar();
		return 0;
	}
	else
	{
		TriclopsError triclops_status;
		TriclopsContext context;
		TriclopsMonoStereoPair monoStereoPair;


		bool saveImages = true;
		bool saveRawImages = true;
		bool saveRectifiedImages = true;
		bool saveDisparityImage = false;
		bool saveOpenCVDisparityImage = false;
		bool showImages = false;

		if (saveRawImages || saveRectifiedImages || saveDisparityImage || saveOpenCVDisparityImage)
			saveImages = true;

		std::string fname_image_left, fname_image_right, fname_image_left_rectified, fname_image_right_rectified, fname_image_disparity, fname_opencv_image_disparity, fname_timestamp;
		int recordTimeInSeconds = 60;

		bool wideMode = true;
		int numDisp = 64;
		int blockSize = 19;
		Bumblebee bParam;
		Bumblebee *bumblebeeObject = new Bumblebee(wideMode, bParam.Res_1280X960, numDisp, blockSize);

		string mode;
		if (wideMode)
			mode = "wide";
		else
			mode = "narrow";

		string res = getStr(bumblebeeObject->outputWidth) + "X" + getStr(bumblebeeObject->outputHeight);

		Global_LeftIMG_Dir = img_root_dir + "/" + seq + "_" + mode + "_res" + res + "_" + getStr(bumblebeeObject->SGBM_Params.numDisp) + "bit_" + getStr(bumblebeeObject->SGBM_Params.blockSize) + "b";
		std::cout << "SEQ:" << seq << endl;
		Global_RightIMG_Dir = img_root_dir + "/" + seq + "_" + mode + "_res" + res + "_" + getStr(bumblebeeObject->SGBM_Params.numDisp) + "bit_" + getStr(bumblebeeObject->SGBM_Params.blockSize) + "b";
		Global_DisparityIMG_Dir = img_root_dir + "/" + seq + "_" + mode + "_res" + res + "_" + getStr(bumblebeeObject->SGBM_Params.numDisp) + "bit_" + getStr(bumblebeeObject->SGBM_Params.blockSize) + "b";

		fname_timestamp = Global_LeftIMG_Dir + "/timeStamps.txt";
		ofstream mytimestampfile(fname_timestamp, ios::app);
		std::cout << "global: "<<Global_LeftIMG_Dir << std::endl;
		createDirectoryIf(Global_LeftIMG_Dir);

		int nrows, newrows, nSrcRows, nSrcCols;
		int ncols, newcols;

		// camera to be used to grab color image
		FC2::Camera camera;
		// grabbed unprocessed image
		FC2::Image grabbedImage;

		std::cout << "Start Connection" << std::endl;

		// connect camera
		FC2::Error flycap_status = camera.Connect();
		FC2T::ErrorType fc2TriclopsError;
		fc2TriclopsError = FC2T::setStereoMode(camera, bumblebeeObject->cameraStereoMode);

		FC2::CameraInfo camInfo;
		FC2::Error fc2Error = camera.GetCameraInfo(&camInfo);
		fc2TriclopsError = FC2T::getContextFromCamera(camInfo.serialNumber, &context);


		//triclopsGetResolution(context, &nrows, &ncols);
		newcols = bumblebeeObject->outputWidth;
		newrows = bumblebeeObject->outputHeight;

		triclops_status = triclopsSetResolution(context, newrows, newcols); //triclopsPrepareRectificationData instead
		triclops_status = triclopsGetResolution(context, &nrows, &ncols);
		triclops_status = triclopsGetSourceResolution(context, &nSrcRows, &nSrcCols);
		triclops_status = triclopsSetDoStereo(context, true);
		triclops_status = triclopsSetDisparity(context, bumblebeeObject->SGBM_Params.minDisp, bumblebeeObject->SGBM_Params.numDisp);

		int w = round(newcols / 4);
		int h = round(newrows / 4);

		if (showImages)
		{
			/// Create windows
			namedWindow("Left Image Rectified", CV_WINDOW_NORMAL);
			resizeWindow("Left Image Rectified", w, h);
			cv::moveWindow("Left Image Rectified", 10, 50);

			namedWindow("Disparity Image", CV_WINDOW_NORMAL);
			resizeWindow("Disparity Image", newcols, newrows);
			cv::moveWindow("Disparity Image", w + 250, 50);			

		}

		std::cout << "Start Capture" << std::endl;
		fc2Error = camera.StartCapture();

		const char* errC = fc2Error.GetDescription();
		std::string err = errC;

		if (strcmp(err.c_str(), "Ok.") != 0)
		{
			cout << "Camera Could not Open !!!: " << err.c_str() << endl << endl;
		}

		//=============================================================================
		// Parameter setting
		//=============================================================================		

		triclops_status = triclopsSetCameraConfiguration(context, bumblebeeObject->cameraStereoCalculationMode);
		triclops_status = triclopsSetStereoAlgorithm(context, bumblebeeObject->SGBM_Params.selectedAlgorithm_LeftREF);
		triclops_status = triclopsSetResolution(context, bumblebeeObject->outputHeight, bumblebeeObject->outputWidth);
		triclops_status = triclopsSetOpenCVDisparityRange(context, bumblebeeObject->SGBM_Params.minDisp, bumblebeeObject->SGBM_Params.numDisp);
		triclops_status = triclopsSetOpenCVStereoMaskSize(context, bumblebeeObject->SGBM_Params.blockSize);
		triclops_status = triclopsSetOpenCVPreFilterCap(context, bumblebeeObject->SGBM_Params.prefilterCap);
		triclops_status = triclopsSetOpenCVSpeckleWindowSize(context, bumblebeeObject->SGBM_Params.speckleSize);
		triclops_status = triclopsSetOpenCVSpeckleRange(context, bumblebeeObject->SGBM_Params.speckleRange);
		triclops_status = triclopsSetOpenCVUniquenessRatio(context, bumblebeeObject->SGBM_Params.uniquenessRatio);
		// OpenCV Stero Block Matching (BM)-only parameters
		triclops_status = triclopsSetOpenCVBMPreFilterType(context, bumblebeeObject->SGBM_Params.prefilterType);
		triclops_status = triclopsSetOpenCVBMPreFilterSize(context, bumblebeeObject->SGBM_Params.prefilterSize);
		triclops_status = triclopsSetOpenCVBMTextureThreshold(context, bumblebeeObject->SGBM_Params.textureThreshold);
		// OpenCV SGBM parameters
		triclops_status = triclopsSetOpenCVSGBMP1(context, bumblebeeObject->SGBM_Params.SGBM_P1);
		triclops_status = triclopsSetOpenCVSGBMP2(context, bumblebeeObject->SGBM_Params.SGBM_P2);
		triclops_status = triclopsSetOpenCVSGBMMode(context, bumblebeeObject->SGBM_Params.SGBM_Mode);

		int index = 9;
		int index2 = 0;

		diff_disparity = 0.0;
		double total_diff_disparity = 0.0;


		// ------------------------------ Start Process Once ---------------------------------------------------------- //
		fc2Error = camera.RetrieveBuffer(&grabbedImage);
		errC = fc2Error.GetDescription();
		err = errC;

		ImageContainer imageCont;
		generatemonoStereoPair(context, grabbedImage, imageCont, monoStereoPair);

		triclops_status = triclopsPrepareRectificationData(context, bumblebeeObject->outputHeight, bumblebeeObject->outputWidth, monoStereoPair.right.nrows, monoStereoPair.right.ncols);//could be beneficial in multi CPU code or paralel computing
																																														 //Rectify
		triclops_status = triclopsRectify(context, &monoStereoPair);

		TriclopsImage imageL, imageR;
		triclops_status = triclopsGetImage(context, TriImg_RECTIFIED, TriCam_RIGHT, &imageR);
		triclops_status = triclopsGetImage(context, TriImg_RECTIFIED, TriCam_LEFT, &imageL);

		TriclopsImage16 imageDisparity16;
		triclops_status = triclopsStereo(context);
		triclops_status = triclopsGetImage16(context, TriImg16_DISPARITY, TriclopsCamera::TriCam_REFERENCE, &imageDisparity16);
		// ------------------------------ Start Process Once ---------------------------------------------------------- //

		t_start = 0.0;
		t_end = 0.0;
		while (strcmp(err.c_str(), "Ok.") == 0 /*&& diff_disparity <= recordTimeInSeconds * 1000*/)
		{
			diff_disparity = (static_cast<float>(t_end) - static_cast<float>(t_start)) / 1000;
			
			///// Clock Start
			t_start = clock();

			total_diff_disparity += diff_disparity;
			cout << index2 << " Dif:" << diff_disparity << "ms. " << "_ FPS:" << (1.0 / diff_disparity) * 1000 << " total:" << total_diff_disparity << " sn." << endl;

			if (mytimestampfile.is_open())
			{
				mytimestampfile << total_diff_disparity << ",";
				mytimestampfile.close();
			}
			else
			{
				mytimestampfile.open(fname_timestamp, ios::app);
				mytimestampfile << total_diff_disparity << ",";
				mytimestampfile.close();
			}

			index = index + 10;
			index2++;

			if (saveImages)
			{
				setImagePaths(index, fname_image_left, fname_image_right, fname_image_left_rectified, fname_image_right_rectified, fname_image_disparity, fname_opencv_image_disparity);
			}

			fc2Error = camera.RetrieveBuffer(&grabbedImage);
			errC = fc2Error.GetDescription();
			err = errC;

			// right and left image extracted from grabbed image
			//ImageContainer imageCont;
			generatemonoStereoPair(context, grabbedImage, imageCont, monoStereoPair);

			//=============================================================================
			// Stereo disparity computation
			//=============================================================================

			// Prepare context for frame rectification
			
			triclops_status = triclopsPrepareRectificationData(context, bumblebeeObject->outputHeight, bumblebeeObject->outputWidth, monoStereoPair.right.nrows, monoStereoPair.right.ncols);//could be beneficial in multi CPU code or paralel computing
			//Rectify
			triclops_status = triclopsRectify(context, &monoStereoPair);

			//TriclopsImage imageL, imageR;
			triclops_status = triclopsGetImage(context, TriImg_RECTIFIED, TriCam_RIGHT, &imageR);
			triclops_status = triclopsGetImage(context, TriImg_RECTIFIED, TriCam_LEFT, &imageL);
			//triclops_examples::handleError("triclopsGetImage()", triclops_status, __LINE__);

			//onur
			cv::Mat imageRight = convertTriclops2OpencvMat(monoStereoPair.right);
			cv::Mat imageLeft = convertTriclops2OpencvMat(monoStereoPair.left);

			cv::Mat imageRight_Rectified = convertTriclops2OpencvMat(imageR);
			cv::Mat imageLeft_Rectified = convertTriclops2OpencvMat(imageL);

			// Do the actual stereo computation
			TriclopsImage16 imageDisparity16;
			triclops_status = triclopsStereo(context);
			triclops_status = triclopsGetImage16(context, TriImg16_DISPARITY, TriclopsCamera::TriCam_REFERENCE, &imageDisparity16);

			if (saveRawImages)
			{
				triclopsSaveImage(&monoStereoPair.left, fname_image_left.c_str());
				triclopsSaveImage(&monoStereoPair.right, fname_image_right.c_str());
				///home/vefak/SANCAKTEPE_TEST_wide_res1280X960_64bit_19b/imgrightRaw000000119.pgm
				
			}

			if (saveRectifiedImages)
			{
				triclopsSaveImage(&imageL, fname_image_left_rectified.c_str());
				triclopsSaveImage(&imageR, fname_image_right_rectified.c_str());
				///home/vefak/SANCAKTEPE_TEST_wide_res1280X960_64bit_19b/imgleft000000079.pgm
				
			}

			if (saveDisparityImage)
			{
				triclopsSaveImage16(&imageDisparity16, fname_image_disparity.c_str());
				///home/vefak/SANCAKTEPE_TEST_wide_res1280X960_64bit_19b/disp000000079.pgm
			}

			cv::Mat imageDisparityMat = convertTriclops2OpencvMat(imageDisparity16);

			double ratio = 1. / 16.;
			auto imgDisparity32F = cv::Mat(newrows, newcols, CV_32F);
			auto imgDisparity8U = cv::Mat(newrows, newcols, CV_8UC1);
			imageDisparityMat.convertTo(imgDisparity8U, CV_8UC1, ratio);
			imageDisparityMat.convertTo(imgDisparity32F, CV_32FC1, ratio);

			if (saveOpenCVDisparityImage)
			{
				imwrite(fname_opencv_image_disparity, imgDisparity8U);
				//home/vefak/SANCAKTEPE_TEST_wide_res1280X960_64bit_19b/disp_opencv000000079.pgm
			}

			float baseline, focallength, centerRow, centerCol;
			triclops_status = triclopsGetBaseline(context, &baseline);
			triclops_status = triclopsGetFocalLength(context, &focallength);//The focal length is in 'pixels' for the current selected output resolution. All cameras' rectified images have the same focal length
			triclops_status = triclopsGetImageCenter(context, &centerRow, &centerCol);

			if (showImages)
			{
				imshow("Disparity Image", imgDisparity8U);
				imshow("Left Image Rectified", imageLeft_Rectified);
				//imshow("Left Image Rectified", imageLeft_Rectified);
				//imshow("Right Image Rectified", imageRight_Rectified);
				//imshow("Left Image", imageLeft);
				//imshow("Right Image", imageRight);
				waitKey(1);
			}


			t_end = clock();

			///// Clock End
		}

		mytimestampfile.close();
		// Close the camera
		camera.StopCapture();
		camera.Disconnect();

		// Destroy the Triclops context
		triclops_status = triclopsDestroyContext(context);
		

		std::cout << "Closed" << std::endl;
	}
	



	getchar();
    return 0;
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

void generatemonoStereoPair(TriclopsContext const &context,FC2::Image const &grabbedImage,	ImageContainer &imageCont,	TriclopsMonoStereoPair &stereoPair)
{
	FC2T::ErrorType fc2TriclopsError;
	TriclopsError te;

	TriclopsImage triclopsImageContainer[2];
	FC2::Image *tmpImage = imageCont.tmp;
	FC2::Image *unprocessedImage = imageCont.unprocessed;

	// Convert the pixel interleaved raw data to de-interleaved and color processed data
	fc2TriclopsError = FC2T::unpackUnprocessedRawOrMono16Image(	grabbedImage,true /*assume little endian*/,	tmpImage[RIGHT], tmpImage[LEFT]);

	//handleError("FC2T::unpackUnprocessedRawOrMono16Image()", fc2TriclopsError, __LINE__);
	unprocessedImage[RIGHT] = tmpImage[RIGHT];
	unprocessedImage[LEFT] = tmpImage[LEFT];

	// create triclops image for right and left lens
	for (size_t i = 0; i < 2; ++i) {
		te = triclopsLoadImageFromBuffer(unprocessedImage[i].GetData(),	unprocessedImage[i].GetRows(),unprocessedImage[i].GetCols(), unprocessedImage[i].GetStride(), &triclopsImageContainer[i]);
		//triclops_examples::handleError("triclopsLoadImageFromBuffer()", te, __LINE__);
	}

	// create stereo input from the triclops images constructed above
	// pack image data into a TriclopsColorStereoPair structure
	te = triclopsBuildMonoStereoPairFromBuffers(context,&triclopsImageContainer[RIGHT],	&triclopsImageContainer[LEFT],&stereoPair);
	//triclops_examples::handleError("triclopsBuildMonoStereoPairFromBuffers()", te, __LINE__);
}

void setImagePaths(int imgNum, std::string &fname_image_left, std::string &fname_image_right, std::string &fname_image_left_rectified, std::string &fname_image_right_rectified, std::string &fname_image_disparity, std::string &fname_openCV_image_disparity)
{
	std::stringstream ss_buffer;
	std::string Global_currentIMGNum_str;

	ss_buffer << imgNum;
	Global_currentIMGNum_str = ss_buffer.str();
	ss_buffer.str(std::string());

	if (imgNum < 100) Global_currentIMGNum_str = "0000000" + Global_currentIMGNum_str;
	if (imgNum > 100) Global_currentIMGNum_str = "000000" + Global_currentIMGNum_str;

	fname_image_left_rectified = Global_LeftIMG_Dir + "/imgleft" + Global_currentIMGNum_str + ".pgm";
	fname_image_right_rectified = Global_RightIMG_Dir + "/imgright" + Global_currentIMGNum_str + ".pgm";
	fname_image_left = Global_LeftIMG_Dir + "/imgleftRaw" + Global_currentIMGNum_str + ".pgm";
	fname_image_right = Global_RightIMG_Dir + "/imgrightRaw" + Global_currentIMGNum_str + ".pgm";
	fname_image_disparity = Global_DisparityIMG_Dir + "/disp" + Global_currentIMGNum_str + ".pgm";
	fname_openCV_image_disparity = Global_DisparityIMG_Dir + "/disp_opencv" + Global_currentIMGNum_str + ".pgm";
	
}

std::string getStr(int def)
{
	std::stringstream ss_buffer;
	std::string return_str;
	ss_buffer << def;
	return_str = ss_buffer.str();
	ss_buffer.str(std::string());
	return return_str;
}

std::string getStr(float def)
{
	std::stringstream ss_buffer;
	std::string return_str;
	ss_buffer << def;
	return_str = ss_buffer.str();
	ss_buffer.str(std::string());
	return return_str;
}

std::string getStr(double def)
{
	std::stringstream ss_buffer;
	std::string return_str;
	ss_buffer << def;
	return_str = ss_buffer.str();
	ss_buffer.str(std::string());
	return return_str;
}


void ImageToWorldCoordinates(MyObject &obj, Bumblebee &bumbleObj)
{
	
	float fx = bumbleObj.stereoParams.f / bumbleObj.stereoParams.sx; //focal_length_pixels in X coordinate
	float fy = bumbleObj.stereoParams.f / bumbleObj.stereoParams.sy; //focal_length_pixels in Y coordinate

	if (obj.d > 0)
	{
		//compute 3D point
		obj.Zs = (fx * bumbleObj.stereoParams.b) / obj.d;
		obj.Xs = (obj.Zs / fx) * (obj.u - bumbleObj.stereoParams.u0);
		//obj.Ys = (obj.Zs / fy) * (stereoParams.v0 - obj.v); //Upline of the object regarding to the camera position (zero (0) if parelel to the camera)
		obj.Ys = (obj.Zs / fy) * (obj.v - bumbleObj.stereoParams.v0);// just for check
		obj.distanceS = sqrt(pow(obj.Xs, 2) + pow(obj.Ys, 2) + pow(obj.Zs, 2));

		// correct with camera position and tilt angle
		obj.Xw = obj.Xs + bumbleObj.stereoParams.cX; // (-b / 2)
		obj.Zw = obj.Zs *cos(bumbleObj.stereoParams.tilt) + obj.Ys *sin(bumbleObj.stereoParams.tilt) + bumbleObj.stereoParams.cZ;
		obj.Yw = -obj.Zs*sin(bumbleObj.stereoParams.tilt) + obj.Ys*cos(bumbleObj.stereoParams.tilt) + bumbleObj.stereoParams.cY;
		obj.distanceW = sqrt(pow(obj.Xw, 2) + pow(obj.Yw, 2) + pow(obj.Zw, 2));		
	}
}

void createDirectoryIf(string strPath)
{

	const int dir_err = mkdir(strPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if ( dir_err == 1)
	{
    		printf("Error creating directory!n\n");
    		exit(1);
	}
}
