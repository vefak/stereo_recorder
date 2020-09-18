#include "stereocore_RT.h"




inline int StereoImageBuffer::getBufferSize() const
{
	return Global_Buffer_Size;
}

inline void StereoImageBuffer::setBufferSize()
{
	if (Global_minIMGIndex == Global_maxIMGIndex) { Global_Buffer_Size = 1; }
	else { Global_Buffer_Size = ((Global_maxIMGIndex - Global_minIMGIndex) / Global_imgIncrement) + 1; }
}

inline void StereoImageBuffer::setDefaultParams()
{

#ifdef WITH_BUMBLEBEE
	if (Global_useBumblebee)
	{
		Bumblebee bParam;
		bool use3Way = false;//for slower systems
		int blocksize = 17;
		bumblebeeObject = new Bumblebee(Global_bumblebee_widemode, bParam.Res_1280X960, Global_numDisparities, blocksize, use3Way);

		setBumblebeeStereoParams(*bumblebeeObject);
	}
#endif

	//Set Default Properties	
	cv::Mat leftIMG_buffer;
	int defTotalRows;
	int defTotalCols;

	if (!Global_useRealTime)
	{
		std::string fname_image_left, fname_image_right;
		getImagePaths(false, Global_minIMGIndex, fname_image_left, fname_image_right);
		leftIMG_buffer = cv::imread(fname_image_left, CV_LOAD_IMAGE_UNCHANGED);

		defTotalRows = leftIMG_buffer.rows;
		defTotalCols = leftIMG_buffer.cols;
	}
	else
	{
		defTotalRows = bumblebeeObject->outputHeight;
		defTotalCols = bumblebeeObject->outputWidth;
	}



	if (Global_CropRectifiedIMG)
	{
		defTotalRows = Global_endV - Global_startV;
		defTotalCols = Global_endU - Global_startU;
	}

	if (Global_imgScale > 0 && Global_imgScale < 1.0)
	{
		Global_imgTotalRows = round(defTotalRows * Global_imgScale);
		Global_imgTotalCols = round(defTotalCols * Global_imgScale);
	}
	else
	{
		Global_imgTotalRows = defTotalRows;
		Global_imgTotalCols = defTotalCols;
	}

	Global_imgMiddleCol = round(Global_imgTotalCols / 2);

	Global_uDisparityGlobals = new uDisparityGlobals();
	Global_vDisparityGlobals = new vDisparityGlobals();
	Global_dDisparityGlobals = new dDisparityGlobals();

	Global_uDisparityParams = new uDisparityParams(Global_numDisparities, Global_imgTotalCols);
	Global_vDisparityParams = new vDisparityParams(Global_imgTotalRows, Global_numDisparities);
	Global_dDisparityParams = new dDisparityParams(Global_imgTotalRows, Global_imgTotalCols);

	//stereoParams = new stereoCamParams();

#ifdef WITH_OCCUPANCY_GRID
	Global_occupancyGridGlobals = new occupancyGridGlobals();
	Global_occupancyGridParams = new occupancyGridParams();
#endif

	
	//Set Default Properties
}

inline void StereoImageBuffer::setImages(bool reverse, bool equalizeHist, bool rectify)//if not using realTime
{
	if (!Global_useRealTime)
	{
		std::string fname_image_left, fname_image_right;
		cv::Mat leftIMG_buffer, leftIMG_buffer_gray, rightIMG_buffer, rightIMG_buffer_gray, leftIMG_pinnedMat, rightIMG_pinnedMat;
		double minValLeft, minValRight; double maxValLeft, maxValRight;
		int img_index = 0;

		std::string mapPath = "E:\\KUDiary\\Important Doc\\3D Detection\\stereo Images\\Bumblebee\\RoomDataCollection\\Occupancy Grid TestBed (edited).bmp";
		mapPath_img = cv::imread(mapPath, CV_LOAD_IMAGE_UNCHANGED);

		for (int imgNnum = Global_minIMGIndex; imgNnum <= Global_maxIMGIndex; imgNnum = imgNnum + Global_imgIncrement)
		{
			getImagePaths(reverse, imgNnum, fname_image_left, fname_image_right);
			Global_currentIMGNum = imgNnum;

			try
			{
				//read gray scale images
				leftIMG_buffer = cv::imread(fname_image_left, CV_LOAD_IMAGE_UNCHANGED);
				rightIMG_buffer = cv::imread(fname_image_right, CV_LOAD_IMAGE_UNCHANGED);

				if (Global_CropRectifiedIMG)
				{
					leftIMG_buffer = leftIMG_buffer(cv::Range(Global_startV, Global_endV), cv::Range(Global_startU, Global_endU));
					rightIMG_buffer = rightIMG_buffer(cv::Range(Global_startV, Global_endV), cv::Range(Global_startU, Global_endU));
				}

				if (Global_imgScale > 0 && Global_imgScale < 1.0)
				{
					cv::resize(leftIMG_buffer, leftIMG_buffer, cv::Size(Global_imgTotalCols, Global_imgTotalRows));
					cv::resize(rightIMG_buffer, rightIMG_buffer, cv::Size(Global_imgTotalCols, Global_imgTotalRows));
				}

				//prepare gray images to convert to 256 bit (UINT8)
				cv::minMaxLoc(leftIMG_buffer, &minValLeft, &maxValLeft);
				cv::minMaxLoc(rightIMG_buffer, &minValRight, &maxValRight);


				if (!Global_useGPU)
				{
					leftIMG_buffer_gray = cv::Mat(Global_imgTotalRows, Global_imgTotalCols, CV_8UC1); //without "new" leftIMG_buffer_gray's ref. remains same
					rightIMG_buffer_gray = cv::Mat(Global_imgTotalRows, Global_imgTotalCols, CV_8UC1); //without "new" rightIMG_buffer_gray's ref. remains same

					if (maxValLeft > 255 || maxValRight > 255)
					{
						//convert gray images to 256 bit (UINT8)
						if (maxValLeft > 255)
						{
							leftIMG_buffer.convertTo(leftIMG_buffer_gray, CV_8UC1, 255 / (maxValLeft - minValLeft));
						}
						else
						{
							leftIMG_buffer.copyTo(leftIMG_buffer_gray);
						}

						if (maxValRight > 255)
						{

							rightIMG_buffer.convertTo(rightIMG_buffer_gray, CV_8UC1, 255 / (maxValRight - minValRight));
						}
						else
						{
							rightIMG_buffer.copyTo(rightIMG_buffer_gray);
						}
					}
					else
					{
						leftIMG_buffer.copyTo(leftIMG_buffer_gray);
						rightIMG_buffer.copyTo(rightIMG_buffer_gray);
					}


					if (equalizeHist)
					{
						cv::equalizeHist(leftIMG_buffer_gray, leftIMG_buffer_gray);
						cv::equalizeHist(rightIMG_buffer_gray, rightIMG_buffer_gray);
					}

					shared_ptr<cv::Mat> sharedptr(new cv::Mat);
					sharedptr->push_back(leftIMG_buffer_gray);

					auto temp_LeftSharedPtrMat = std::make_shared<cv::Mat>(leftIMG_buffer_gray); // or 'std::shared_ptr<int> p' if you insist
					Global_IMG_Left_Image_VectorOfSharedPtr.push_back(temp_LeftSharedPtrMat);

					auto temp_RightSharedPtrMat = std::make_shared<cv::Mat>(rightIMG_buffer_gray); // or 'std::shared_ptr<int> p' if you insist
					Global_IMG_Right_Image_VectorOfSharedPtr.push_back(temp_RightSharedPtrMat);

					img_index++;
				}
#ifdef WITH_GPU
				else
				{
					//Create pinned memory for GPU
					cv::cuda::HostMem *cudaHostMem_Left, *cudaHostMem_Right;

					cudaHostMem_Left = new cv::cuda::HostMem[1]; //without "new" leftIMG_buffer_gray's ref. remains same
					cudaHostMem_Left->alloc_type = cv::cuda::HostMem::AllocType::PAGE_LOCKED;
					cudaHostMem_Right = new cv::cuda::HostMem(cv::cuda::HostMem::AllocType::PAGE_LOCKED);

					cudaHostMem_Left->create(Global_imgTotalRows, Global_imgTotalCols, CV_8UC1);
					cudaHostMem_Right->create(Global_imgTotalRows, Global_imgTotalCols, CV_8UC1);

					leftIMG_pinnedMat = cudaHostMem_Left->createMatHeader();
					rightIMG_pinnedMat = cudaHostMem_Right->createMatHeader();

					if (maxValLeft > 255 || maxValRight > 255)
					{
						//convert gray images to 256 bit (UINT8)
						if (maxValLeft > 255)
						{
							leftIMG_buffer.convertTo(leftIMG_pinnedMat, CV_8UC1, 255 / (maxValLeft - minValLeft));
						}
						else
						{
							leftIMG_buffer.copyTo(leftIMG_pinnedMat);
						}

						if (maxValRight > 255)
						{
							rightIMG_buffer.convertTo(rightIMG_pinnedMat, CV_8UC1, 255 / (maxValRight - minValRight));
						}
						else
						{
							rightIMG_buffer.copyTo(rightIMG_pinnedMat);
						}
					}
					else
					{
						leftIMG_buffer.copyTo(leftIMG_pinnedMat);
						rightIMG_buffer.copyTo(rightIMG_pinnedMat);
					}

					if (equalizeHist)
					{
						cv::equalizeHist(leftIMG_pinnedMat, leftIMG_pinnedMat);
						cv::equalizeHist(rightIMG_pinnedMat, rightIMG_pinnedMat);
					}

					Global_IMG_Left_Image_Buffer_GPU[img_index] = leftIMG_pinnedMat;
					Global_IMG_Right_Image_Buffer_GPU[img_index] = rightIMG_pinnedMat;

					img_index++;
				}
#endif
			}
			catch (cv::Exception &ex)
			{
			std:cout << "Error: " << ex.err << std::endl;
			}
		}
	}
}



inline void StereoImageBuffer::getImagePaths(bool reverse, int imgNum, std::string& fname_image_left, std::string& fname_image_right) const
{
	std::stringstream ss_buffer;
	std::string Global_currentIMGNum_str;

	ss_buffer << imgNum;
	Global_currentIMGNum_str = ss_buffer.str();
	ss_buffer.str(std::string());

	if (imgNum < 100) Global_currentIMGNum_str = "0000000" + Global_currentIMGNum_str;
	if (imgNum > 100) Global_currentIMGNum_str = "000000" + Global_currentIMGNum_str;

	if (!reverse)
	{
		fname_image_left = Global_LeftIMG_Dir + "\\imgleft" + Global_currentIMGNum_str + ".pgm";
		fname_image_right = Global_RightIMG_Dir + "\\imgright" + Global_currentIMGNum_str + ".pgm";
	}
	else
	{
		fname_image_right = Global_LeftIMG_Dir + "\\imgleft" + Global_currentIMGNum_str + ".pgm";
		fname_image_left = Global_RightIMG_Dir + "\\imgright" + Global_currentIMGNum_str + ".pgm";
	}

}


inline void StereoImageBuffer::processImageDisparities(bool showImages)
{
	clock_t t1, t2;
	float diff_disparity;

#ifdef WITH_SLAM
	Global_occupancyGridGlobals->occupancyGridMap.set_OccupancyGridSLAMMatrix(*Global_occupancyGridParams);
#endif

	if (!Global_useRealTime)
	{
#pragma region Offline
		if (!Global_useGPU)//use CPU buffer
		{
			for (int imgindex = 0; imgindex < Global_Buffer_Size; imgindex++)
			{
				Global_currentIMGIndex = imgindex;
				Global_Current_Left_Image = *Global_IMG_Left_Image_VectorOfSharedPtr[imgindex].get();
				Global_Current_Right_Image = *Global_IMG_Right_Image_VectorOfSharedPtr[imgindex].get();


				set_DenseDisparityMap();
				set_UVDisparityMaps();
				//set_BestLine_From_VDisparity();
				//set_BestLine_From_VDisparity_Curve();
				set_BestLine_From_VDisparity_Regression();
				//set_ImageInfinityPoint_And_VehicleRoadLine(Global_imgMiddleCol, Global_vDisparityGlobals->bestRoadLine_vIntersection, Global_imgTotalCols, Global_imgTotalRows);
				set_ImageInfinityPoint_And_VehicleRoadLine(Global_imgMiddleCol, Global_vDisparityGlobals->piecewiseBestLine_RoadEndPoint.y, Global_imgTotalCols, Global_imgTotalRows);

				//cv::Mat uDispDraw1 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, CV_8UC3);
				//cv::Mat uDispDraw2 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, CV_8UC3);
				//cv::Mat uDispDraw3 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, CV_8UC3);
				//cv::Mat uDispDraw4 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, CV_8UC3);
				//cv::Mat uDispDraw5 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, CV_8UC3);

				//diff_disparity = (static_cast<float>(t2)-static_cast<float>(t1));
				//cout << "Time: " << diff_disparity << " ms." << endl;

				//Mat draw0(Global_uDisparityGlobals->uDisparityImage);

				set_LabelsOf_UDisparityLines_Using_LabelingAlg();

				//Mat draw01(Global_uDisparityGlobals->uDisparityImage_thresholded);

				//cv::Mat draw1 = drawUDispObjects(uDispDraw1, Global_labeledObjectList);
				merge_MaskVObjects_From_UDisparity_By_Labeling_Alg(Global_labeledObjectList);
				//cv::Mat draw2 = drawUDispObjects(uDispDraw2, Global_mergedObjectList1);
				merge_InclinedObjects_From_UDisparity_By_Labeling_Alg(Global_mergedObjectList1);
				//cv::Mat draw3 = drawUDispObjects(uDispDraw3, Global_mergedObjectList2);
				set_SideBuildingsIfAny_And_Filter(Global_mergedObjectList2);
				//cv::Mat draw4 = drawUDispObjects(uDispDraw4, Global_mergedObjectList2);
				createDenseDispObjects(Global_mergedObjectList2);
				//cv::Mat draw5= drawUDispObjects(uDispDraw5, Global_mergedObjectList2);

#ifdef WITH_LANE_DETECTION
				laneDetection *detLanes = new laneDetection(1.05, Global_Current_Left_Image);
				detLanes->testLanes(true, Global_Current_Left_Image);
				cv::imshow("Window1", Global_Current_Left_Image);
				int c = (cvWaitKey(1000) & 0xff);
				Mat laneIMG = detLanes->getLanes(false, Global_vDisparityGlobals->bestRoadLine_vIntersection);
#endif			


				//cv::Mat draw4 = drawUDispObjects(uDispDraw4, Global_mergedObjectList2);				

				cv::Mat colorV;
				cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV, CV_GRAY2BGR);
				//cv::line(colorV, Point(0, Global_vDisparityGlobals->bestRoadLine_vIntersection), Point(Global_vDisparityGlobals->bestRoadLine_dInterSection, Global_vDisparityParams->totalRows - 1), cv::Scalar(255, 0, 255));

				if (Global_vDisparityGlobals->bestRoadLine_a == 0)
				{
					cv::line(colorV, Point(Global_vDisparityGlobals->bestRoadLine[0], Global_vDisparityGlobals->bestRoadLine[1]), Point(Global_vDisparityGlobals->bestRoadLine[2], Global_vDisparityGlobals->bestRoadLine[3]), cv::Scalar(0, 0, 255));
					cv::line(colorV, Point(Global_vDisparityGlobals->piecewiseBestLine[0], Global_vDisparityGlobals->piecewiseBestLine[1]), Point(Global_vDisparityGlobals->piecewiseBestLine[2], Global_vDisparityGlobals->piecewiseBestLine[3]), cv::Scalar(0, 255, 0));
					circle(colorV, Global_vDisparityGlobals->piecewiseBestLine_RoadEndPoint, 3, Scalar(255, 255, 0), 2);
					circle(colorV, Global_vDisparityGlobals->piecewiseBestLine_RoadStartPoint, 3, Scalar(255, 255, 0), 2);
				}
				else
				{
					std::vector<Point2f> curvePoints;

					int start_point_x = 0;
					int end_point_x = Global_vDisparityParams->totalCols - 1;
					for (float x = start_point_x; x <= end_point_x; x += 1) {
						float y = Global_vDisparityGlobals->bestRoadLine_m*x + Global_vDisparityGlobals->bestRoadLine_a*x*x + Global_vDisparityGlobals->bestRoadLine_b;
						Point2f new_point = Point2f(x, y);                  //resized to better visualize
						curvePoints.push_back(new_point);
					}

					Mat curve(curvePoints, true);
					curve.convertTo(curve, CV_32S); //adapt type for polylines

					cv::line(colorV, Point(Global_vDisparityGlobals->piecewiseBestLine[0], Global_vDisparityGlobals->piecewiseBestLine[1]), Point(Global_vDisparityGlobals->piecewiseBestLine[2], Global_vDisparityGlobals->piecewiseBestLine[3]), cv::Scalar(0, 255, 0));
					polylines(colorV, curve, false, Scalar(255), 2, CV_AA);
					circle(colorV, Global_vDisparityGlobals->piecewiseBestLine_RoadEndPoint, 3, Scalar(255, 255, 0), 2);
					circle(colorV, Global_vDisparityGlobals->piecewiseBestLine_RoadStartPoint, 3, Scalar(255, 255, 0), 2);
				}

				//int v = get_v_From_VDisparity_RoadProfile(24);
				cv::Mat dispV = Global_vDisparityGlobals->vDisparityImage;
				cv::Mat dispU = Global_uDisparityGlobals->uDisparityImage;
				cv::Mat dispUT = Global_uDisparityGlobals->uDisparityImage_thresholded;
				cv::Mat dispD = Global_Current_DenseDisparity_Image;
				//cv::Mat drawed = drawObjects(Global_Current_Left_Image, Global_mergedObjectList2);

				cv::Mat drawed;
				cv::cvtColor(Global_Current_Left_Image, drawed, CV_GRAY2RGB);

				cv::Mat uDispDraw = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, drawed.type());
				uDispDraw = drawUDispObjects(uDispDraw, Global_mergedObjectList2);

				//int deleted = getNumObjectsDeleted(Global_mergedObjectList2);
				//int livingObj = Global_currentNumOfObjects - deleted;
				//diff_disparity = (static_cast<float>(t2)-static_cast<float>(t1));
				//cout << "NumOfObjs:" << Global_currentNumOfObjects << " LivingObjects:" << livingObj << " time: " << diff_disparity << "ms." << endl;

#ifdef WITH_OCCUPANCY_GRID				
				set_occupancyUDisparityMap();
#ifdef WITH_SLAM
				//if (Global_occupancyGridParams->useJointProbilities)//delete later, just for testing
				//{
				//	Mat check_PCu, check_PVu, check_POu;
				//	check_POu = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix;
				//	check_PCu = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixPCu;
				//	check_PVu = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixPVu;
				//	cout << "";
				//}
#endif

				//Global_occupancyGridGlobals->occupancyGridMap.
				//Global_occupancyGridGlobals->occupancyGridMap.print_OccupancyGridMatrix();
#ifdef WITH_SLAM_TEST
				//centerSLAM.x = Global_occupancyGridParams->middleCol;// round((Global_occupancyGridGlobals->occupancyGridMap.middleColSLAM + Global_occupancyGridGlobals->occupancyGridMap.middleRowSLAM + Global_occupancyGridGlobals->occupancyGridMap.middleCol + Global_occupancyGridGlobals->occupancyGridMap.middleRow) / 2.0);// Global_occupancyGridGlobals->occupancyGridMap.middleCol; //600; //(double)Global_occupancyGridGlobals->occupancyGridMap.middleCol;
				//centerSLAM.y = Global_occupancyGridParams->totalRows;// round(((Global_occupancyGridGlobals->occupancyGridMap.middleRowSLAM + Global_occupancyGridGlobals->occupancyGridMap.middleRow) - (Global_occupancyGridGlobals->occupancyGridMap.middleColSLAM - Global_occupancyGridGlobals->occupancyGridMap.middleCol)) / 2.0); //Global_occupancyGridGlobals->occupancyGridMap.middleColSLAM;
				//centerSLAM.x = Global_occupancyGridGlobals->occupancyGridMap.middleColSLAM;
				//centerSLAM.y = Global_occupancyGridGlobals->occupancyGridMap.totalRowsSL;

				/*
				int noOfObservations = 11;
				float NpUD = 100;
				float* NvUD = new float[noOfObservations] { 100, 90, 50, 50, 0, 100, 80, 70, 50, 100, 100};
				float* NoUD = new float[noOfObservations] { 100, 90, 50, 40, 0, 70, 60, 50, 10, 90, 100};
				float PVu, RoUD, PCu, POu, POu2;
				float aprioProbility = 0.5, aprioProbility2 = 1.0;
				Global_occupancyGridParams->uDisp_probabilityFP = 0.0;
				Global_occupancyGridParams->uDisp_probabilityFN = 0.0;

				float PA, PB, PAB, PBA;

				for (int expNum = 0; expNum < noOfObservations; expNum++)
				{
				PVu = NvUD[expNum] / NpUD;

				if (NvUD[expNum] > 0)
				RoUD = NoUD[expNum] / NvUD[expNum];
				else
				RoUD = 0;

				//PCu = 1 - exp(-(RoUD / Global_occupancyGridParams->uDisp_tO));
				PCu = RoUD;

				POu = PVu*PCu*(1 - Global_occupancyGridParams->uDisp_probabilityFP) + PVu*(1 - PCu)*Global_occupancyGridParams->uDisp_probabilityFN + (1 - PVu)*aprioProbility;
				POu2 = PVu*PCu*(1 - Global_occupancyGridParams->uDisp_probabilityFP) + PVu*(1 - PCu)*Global_occupancyGridParams->uDisp_probabilityFN + (1 - PVu)*aprioProbility2;
				float orgPOu = PVu*PCu*(1 - Global_occupancyGridParams->uDisp_probabilityFP) + PVu*(1 - PCu)*Global_occupancyGridParams->uDisp_probabilityFN + (1 - PVu)*0.5;

				cout << fixed << std::setprecision(2) << "AprioP:" << aprioProbility << " FPR:" << Global_occupancyGridParams->uDisp_probabilityFP << " FNR:" << Global_occupancyGridParams->uDisp_probabilityFN << " Np:" << NpUD << " Nv:" << NvUD[expNum] << " No:" << NoUD[expNum] << " --> PVu:" << PVu << " Ro:" << RoUD << " PCu:" << PCu << " POu:" << POu <<  " orgPOu:" << orgPOu << endl;

				PA = aprioProbility2;
				PBA = POu2;
				PB = orgPOu;

				PAB = (PBA*PA) / PB;

				cout << fixed << std::setprecision(2) << "PA (AprioP):" << PA << " PBA (new POU2):" << PBA << " PB (org PB):" << PB << " PAB (Result):" << PAB << endl << endl;

				aprioProbility = POu;
				aprioProbility2 = PBA;
				}
				*/
				Point2f centerSLAM;
				//centerSLAM.x = Global_occupancyGridParams->middleColSLAM - Global_occupancyGridParams->middleCol;
				//centerSLAM.y = Global_occupancyGridParams->middleRowSLAM - Global_occupancyGridParams->totalRows;

				centerSLAM.x = Global_occupancyGridParams->middleCol;
				centerSLAM.y = Global_occupancyGridParams->totalRows;
				float tx = Global_occupancyGridParams->middleColSLAM;
				float ty = Global_occupancyGridParams->middleRowSLAM;

				//float tx = round(Global_occupancyGridParams->middleColSLAM - centerSLAM.x);
				//float ty = round(Global_occupancyGridParams->middleRowSLAM - centerSLAM.y);

				double rotateDegree = 0;

				//print rotate matrix
				Mat M = getRotationMatrix2D(centerSLAM, rotateDegree, 1);
				Mat MI;
				invertAffineTransform(M, MI);


				Mat t = cv::Mat::zeros(2, 3, CV_64FC1); //clockwise //To represent affine transformations with matrices, we can use homogeneous coordinates.This means representing a 2 - vector(x, y) as a 3 - vector(x, y, 1),

				t.at<double>(0, 0) = cos(rotateDegree * CV_PI / 180.0);//cos(teta)
				t.at<double>(0, 1) = sin(rotateDegree * CV_PI / 180.0);//sin(teta)
				t.at<double>(0, 2) = centerSLAM.y;//y trans
				t.at<double>(1, 0) = -sin(rotateDegree * CV_PI / 180.0);//-sin(teta)
				t.at<double>(1, 1) = cos(rotateDegree * CV_PI / 180.0);//cosd(teta)
				t.at<double>(1, 2) = centerSLAM.x;//x trans

												  //cout << "M" << endl;

												  //for (int r = 0; r < M.rows; r++)
												  //{
												  //	for (int c = 0; c < M.cols; c++)
												  //	{
												  //		cout << fixed << std::setprecision(2) << M.at<double>(r,c) << "  ";
												  //	}
												  //	cout << endl;
												  //}

												  //cout << "MI" << endl;

												  //for (int r = 0; r < M.rows; r++)
												  //{
												  //	for (int c = 0; c < M.cols; c++)
												  //	{
												  //		cout << fixed << std::setprecision(2) << MI.at<double>(r, c) << "  ";
												  //	}
												  //	cout << endl;
												  //}

												  //cout << "M_Manual" << endl;

												  //for (int r = 0; r < M.rows; r++)
												  //{
												  //	for (int c = 0; c < M.cols; c++)
												  //	{
												  //		cout << fixed << std::setprecision(2) << t.at<double>(r, c) << "  ";
												  //	}
												  //	cout << endl;
												  //}


				Mat dst0 = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix;
				Mat dst1 = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM;
				//Mat dst_manual = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM;
				Mat dst_manual = Mat::zeros(Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM.rows, Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM.cols, CV_32FC1);

				Mat dst;
				warpAffine(Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix, dst, M, Size(Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM.cols, Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM.rows));
				//warpAffine(Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix, dst, M, Size(Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.cols, Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.rows));


				bool bilinearInterpolation = true;
				bool bilinearInterpolationExtended = true;

				float x_real, y_real, x_real2, y_real2, x_real3, y_real3;
				int x, y;
				float ct, rt;

				//find boundaries not to search all pixels to interpolate
				float *xPoints = new float[4];
				float *yPoints = new float[4];

				//[0,0]
				float rectLeftUpPointX = 0, rectLeftUpPointX_map;
				float rectLeftUpPointY = 0, rectLeftUpPointY_map;
				ct = (rectLeftUpPointX - centerSLAM.x);// (c - origin_x)
				rt = (rectLeftUpPointY - centerSLAM.y);//(r - origin_y)
				rectLeftUpPointX_map = ct*M.at<double>(0, 0) + rt*M.at<double>(0, 1) + tx;
				rectLeftUpPointY_map = ct*M.at<double>(1, 0) + rt*M.at<double>(1, 1) + ty;
				xPoints[0] = rectLeftUpPointX_map;
				yPoints[0] = rectLeftUpPointY_map;

				//[0,c]
				float rectRightUpPointX = Global_occupancyGridParams->totalCols, rectRightUpPointX_map;
				float rectRightUpPointY = 0, rectRightUpPointY_map;
				ct = (rectRightUpPointX - centerSLAM.x);// (c - origin_x)
				rt = (rectRightUpPointY - centerSLAM.y);//(r - origin_y)
				rectRightUpPointX_map = ct*M.at<double>(0, 0) + rt*M.at<double>(0, 1) + tx;
				rectRightUpPointY_map = ct*M.at<double>(1, 0) + rt*M.at<double>(1, 1) + ty;
				xPoints[1] = rectRightUpPointX_map;
				yPoints[1] = rectRightUpPointY_map;

				//[r,0]
				float rectLeftDownPointX = 0, rectLeftDownPointX_map;
				float rectLeftDownPointY = Global_occupancyGridParams->totalRows, rectLeftDownPointY_map;
				ct = (rectLeftDownPointX - centerSLAM.x);// (c - origin_x)
				rt = (rectLeftDownPointY - centerSLAM.y);//(r - origin_y)
				rectLeftDownPointX_map = ct*M.at<double>(0, 0) + rt*M.at<double>(0, 1) + tx;
				rectLeftDownPointY_map = ct*M.at<double>(1, 0) + rt*M.at<double>(1, 1) + ty;
				xPoints[2] = rectLeftDownPointX_map;
				yPoints[2] = rectLeftDownPointY_map;

				//[r,c]
				float rectRightDownPointX = Global_occupancyGridParams->totalCols, rectRightDownPointX_map;
				float rectRightDownPointY = Global_occupancyGridParams->totalRows, rectRightDownPointY_map;
				ct = (rectRightDownPointX - centerSLAM.x);// (c - origin_x)
				rt = (rectRightDownPointY - centerSLAM.y);//(r - origin_y)
				rectRightDownPointX_map = ct*M.at<double>(0, 0) + rt*M.at<double>(0, 1) + tx;
				rectRightDownPointY_map = ct*M.at<double>(1, 0) + rt*M.at<double>(1, 1) + ty;
				xPoints[3] = rectRightDownPointX_map;
				yPoints[3] = rectRightDownPointY_map;

				//find the smallest and biggest x and y points to draw rectangle boundaries to search for interpolation

				//for rows (x Points)
				float rStart = Global_occupancyGridParams->totalRowsSLAM;
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

				if (rEnd > Global_occupancyGridParams->totalRowsSLAM)
					rEnd = Global_occupancyGridParams->totalRowsSLAM;

				//for columns (y Points)
				float cStart = Global_occupancyGridParams->totalColsSLAM;
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

				if (cEnd > Global_occupancyGridParams->totalColsSLAM)
					cEnd = Global_occupancyGridParams->totalColsSLAM;

				int rStarti = (int)round(rStart);
				int cStarti = (int)round(cStart);
				int rEndi = (int)round(rEnd);
				int cEndi = (int)round(cEnd);
				//find boundaries not to search all pixels to interpolate

				if (bilinearInterpolationExtended)
				{
					for (int r = rStarti; r < rEndi; r++)
					{
						for (int c = cStarti; c < cEndi; c++)
						{
							ct = (c - tx);// (c - origin_x)
							rt = (r - ty);//(r - origin_y)

							x_real = ct*MI.at<double>(0, 0) + rt*MI.at<double>(0, 1) + centerSLAM.x;
							y_real = ct*MI.at<double>(1, 0) + rt*MI.at<double>(1, 1) + centerSLAM.y;

							//x_real = c*MI.at<double>(0, 0) + r*MI.at<double>(0, 1) + MI.at<double>(0, 2);
							//y_real = c*MI.at<double>(1, 0) + r*MI.at<double>(1, 1) + MI.at<double>(1, 2);


							x_real2 = c*MI.at<double>(0, 0) + r*MI.at<double>(0, 1) + MI.at<double>(0, 2);
							y_real2 = c*MI.at<double>(1, 0) + r*MI.at<double>(1, 1) + MI.at<double>(1, 2);

							float x2 = y_real;
							float y2 = x_real;

							float u = floor(x2);
							float v = floor(y2);

							float W1 = (u + 1 - x2)*(v + 1 - y2);
							float W2 = (x2 - u)*(v + 1 - y2);
							float W3 = (u + 1 - x2)*(y2 - v);
							float W4 = (x2 - u)*(y2 - v);

							if ((u >= 0 && u <  Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.rows - 1) && (v >= 0 && v <  (Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.cols - 1)))
							{
								float res = W1* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v) + W2* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v) + W3* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v + 1) + W4* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v + 1);
								dst_manual.at<float>(r, c) = res;
							}
							else
							{
								//dst_manual.at<float>(r, c) = 0.5;
							}
						}
					}


					rotateDegree = -22.5;

					//print rotate matrix
					M = getRotationMatrix2D(centerSLAM, rotateDegree, 1);
					MI;
					invertAffineTransform(M, MI);

					for (int r = 0; r < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM.rows; r++)
					{
						for (int c = 0; c < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM.cols; c++)
						{
							//float ct = (c - centerSLAM.x);// (c - origin_x)
							//float rt = (r - centerSLAM.y);//(r - origin_y)

							float ct = (c - tx);// (c - origin_x)
							float rt = (r - ty);//(r - origin_y)

							x_real = ct*MI.at<double>(0, 0) + rt*MI.at<double>(0, 1) + centerSLAM.x;
							y_real = ct*MI.at<double>(1, 0) + rt*MI.at<double>(1, 1) + centerSLAM.y;

							//x_real = c*MI.at<double>(0, 0) + r*MI.at<double>(0, 1) + MI.at<double>(0, 2);
							//y_real = c*MI.at<double>(1, 0) + r*MI.at<double>(1, 1) + MI.at<double>(1, 2);


							x_real2 = c*MI.at<double>(0, 0) + r*MI.at<double>(0, 1) + MI.at<double>(0, 2);
							y_real2 = c*MI.at<double>(1, 0) + r*MI.at<double>(1, 1) + MI.at<double>(1, 2);

							float x2 = y_real;
							float y2 = x_real;

							float u = floor(x2);
							float v = floor(y2);

							float W1 = (u + 1 - x2)*(v + 1 - y2);
							float W2 = (x2 - u)*(v + 1 - y2);
							float W3 = (u + 1 - x2)*(y2 - v);
							float W4 = (x2 - u)*(y2 - v);

							if ((u >= 0 && u <  Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.rows - 1) && (v >= 0 && v <  (Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.cols - 1)))
							{
								float res = W1* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v) + W2* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v) + W3* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v + 1) + W4* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v + 1);
								dst_manual.at<float>(r, c) = res;
							}
							else
							{
								//dst_manual.at<float>(r, c) = 0.5;
							}
						}
					}


					//rotateDegree = -75;

					////print rotate matrix
					//M = getRotationMatrix2D(centerSLAM, rotateDegree, 1);
					//MI;
					//invertAffineTransform(M, MI);

					//for (int r = 0; r < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM.rows; r++)
					//{
					//	for (int c = 0; c < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM.cols; c++)
					//	{
					//		float ct = (c - centerSLAM.x);// (c - origin_x)
					//		float rt = (r - centerSLAM.y);//(r - origin_y)

					//		x_real = ct*MI.at<double>(0, 0) + rt*MI.at<double>(0, 1) + centerSLAM.x;
					//		y_real = ct*MI.at<double>(1, 0) + rt*MI.at<double>(1, 1) + centerSLAM.y;

					//		float x2 = y_real;
					//		float y2 = x_real;

					//		float u = floor(x2);
					//		float v = floor(y2);

					//		float W1 = (u + 1 - x2)*(v + 1 - y2);
					//		float W2 = (x2 - u)*(v + 1 - y2);
					//		float W3 = (u + 1 - x2)*(y2 - v);
					//		float W4 = (x2 - u)*(y2 - v);

					//		if ((u >= 0 && u <  Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.rows - 1) && (v >= 0 && v <  Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.cols - 1))
					//		{
					//			float res = W1* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v) + W2* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v) + W3* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v + 1) + W4* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v + 1);

					//			if (dst_manual.at<float>(r + ty, c + tx) == 0 || dst_manual.at<float>(r + ty, c + tx) == 0.5)
					//				dst_manual.at<float>(r + ty, c + tx) = res;
					//			else
					//			{
					//				dst_manual.at<float>(r + ty, c + tx) = (dst_manual.at<float>(r + ty, c + tx) + res) / 2.0;
					//			}
					//		}
					//	}

					//}

					//rotateDegree = -90;

					////print rotate matrix
					//M = getRotationMatrix2D(centerSLAM, rotateDegree, 1);
					//MI;
					//invertAffineTransform(M, MI);

					//for (int r = 0; r < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM.rows; r++)
					//{
					//	for (int c = 0; c < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM.cols; c++)
					//	{
					//		float ct = (c - centerSLAM.x);// (c - origin_x)
					//		float rt = (r - centerSLAM.y);//(r - origin_y)

					//		x_real = ct*MI.at<double>(0, 0) + rt*MI.at<double>(0, 1) + centerSLAM.x;
					//		y_real = ct*MI.at<double>(1, 0) + rt*MI.at<double>(1, 1) + centerSLAM.y;

					//		float x2 = y_real;
					//		float y2 = x_real;

					//		float u = floor(x2);
					//		float v = floor(y2);

					//		float W1 = (u + 1 - x2)*(v + 1 - y2);
					//		float W2 = (x2 - u)*(v + 1 - y2);
					//		float W3 = (u + 1 - x2)*(y2 - v);
					//		float W4 = (x2 - u)*(y2 - v);

					//		if ((u >= 0 && u <  Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.rows - 1) && (v >= 0 && v <  Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.cols - 1))
					//		{
					//			float res = W1* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v) + W2* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v) + W3* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v + 1) + W4* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v + 1);

					//			if (dst_manual.at<float>(r + ty, c + tx) == 0 || dst_manual.at<float>(r + ty, c + tx) == 0.5)
					//				dst_manual.at<float>(r + ty, c + tx) = res;
					//			else
					//			{
					//				dst_manual.at<float>(r + ty, c + tx) = (dst_manual.at<float>(r + ty, c + tx) + res) / 2.0;
					//			}
					//		}
					//	}

					//}

					//cout << "hey";
				}
				else
				{
					//centerSLAM.x = round(Global_occupancyGridParams->middleColSLAM - Global_occupancyGridParams->middleCol);
					//centerSLAM.y = round(Global_occupancyGridParams->middleRowSLAM - Global_occupancyGridParams->totalRows);

					//centerSLAM.x = Global_occupancyGridParams->middleColSLAM;
					//centerSLAM.y = Global_occupancyGridParams->middleRowSLAM;

					rotateDegree = 0;
					centerSLAM.x = Global_occupancyGridParams->middleCol;
					centerSLAM.y = Global_occupancyGridParams->totalRows;
					tx = Global_occupancyGridParams->middleColSLAM;
					ty = Global_occupancyGridParams->middleRowSLAM;
					M = getRotationMatrix2D(centerSLAM, rotateDegree, 1);

					for (int r = 0; r < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.rows; r++)
					{
						for (int c = 0; c < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.cols; c++)
						{
							float ct = (c - centerSLAM.x);// (c - origin_x)
							float rt = (r - centerSLAM.y);//(r - origin_y)

							x_real2 = c*M.at<double>(0, 0) + r*M.at<double>(0, 1) + M.at<double>(0, 2);
							y_real2 = c*M.at<double>(1, 0) + r*M.at<double>(1, 1) + M.at<double>(1, 2);

							x_real = ct*t.at<double>(0, 0) + rt*t.at<double>(0, 1) + tx;
							y_real = ct*t.at<double>(1, 0) + rt*t.at<double>(1, 1) + ty;

							x_real3 = c*t.at<double>(0, 0) + r*t.at<double>(0, 1) + centerSLAM.x;
							y_real3 = c*t.at<double>(1, 0) + r*t.at<double>(1, 1) + centerSLAM.y;

							x = (int)round(x_real);//Nearest Neighbourhood interpolation
							y = (int)round(y_real);//Nearest Neighbourhood interpolation

							if (!bilinearInterpolation)
							{
								if (y >= 0 && y < dst_manual.rows)
								{
									if (x >= 0 && x < dst_manual.cols)
									{
										if (Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(r, c) > 0)
										{
											dst_manual.at<float>(y, x) = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(r, c);
											//cout << fixed << std::setprecision(2) << "(" << y << "," << x << ") -->" << Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(r, c) << "  " << endl;
										}
										else
										{
											//dst_manual.at<float>(y, x) = 0.5;
										}
									}
								}
							}
							else
							{
								x_real = ct*MI.at<double>(0, 0) + rt*MI.at<double>(0, 1) + centerSLAM.x;
								y_real = ct*MI.at<double>(1, 0) + rt*MI.at<double>(1, 1) + centerSLAM.y;

								float x2 = y_real;
								float y2 = x_real;

								float u = floor(x2);
								float v = floor(y2);

								float W1 = (u + 1 - x2)*(v + 1 - y2);
								float W2 = (x2 - u)*(v + 1 - y2);
								float W3 = (u + 1 - x2)*(y2 - v);
								float W4 = (x2 - u)*(y2 - v);

								if ((u >= 0 && u < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.rows - 1) && (v >= 0 && v < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.cols - 1))
								{
									float res = W1* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v) + W2* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v) + W3* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v + 1) + W4* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v + 1);
									dst_manual.at<float>(r + ty, c + tx) = res;
								}
								else
								{
									dst_manual.at<float>(r + ty, c + tx) = 0.5;
								}
							}
							//t = [cosd(teta) - sind(teta); sind(teta) cosd(teta)]; //matlab code
							//affine = [xscale*cosd(teta) xshear*sind(teta) xtrans; yshear*(-sind(teta)) yscale*cosd(teta) ytrans; 0 0 1]; //matlab code
							//xyz = affine*[(r - origin_x); (c - origin_y); 1]; //matlab code
							//xy = t*[(r - origin_x); (c - origin_y)];

						}
					}

					rotateDegree = 22.5;
					M = getRotationMatrix2D(centerSLAM, rotateDegree, 1);
					centerSLAM.x = Global_occupancyGridParams->middleCol;
					centerSLAM.y = Global_occupancyGridParams->middleRow;
					tx = Global_occupancyGridParams->middleColSLAM;
					ty = Global_occupancyGridParams->middleRowSLAM;
					M = getRotationMatrix2D(centerSLAM, rotateDegree, 1);

					t.at<double>(0, 0) = cos(rotateDegree * CV_PI / 180.0);//cos(teta)
					t.at<double>(0, 1) = sin(rotateDegree * CV_PI / 180.0);//sin(teta)
					t.at<double>(0, 2) = centerSLAM.y;//y trans
					t.at<double>(1, 0) = -sin(rotateDegree * CV_PI / 180.0);//-sin(teta)
					t.at<double>(1, 1) = cos(rotateDegree * CV_PI / 180.0);//cosd(teta)
					t.at<double>(1, 2) = centerSLAM.x;//x trans

					for (int r = 0; r < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.rows; r++)
					{
						for (int c = 0; c < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.cols; c++)
						{
							float ct = (c - centerSLAM.x);// (c - origin_x)
							float rt = (r - centerSLAM.y);//(r - origin_y)

							x_real2 = c*M.at<double>(0, 0) + r*M.at<double>(0, 1) + M.at<double>(0, 2);
							y_real2 = c*M.at<double>(1, 0) + r*M.at<double>(1, 1) + M.at<double>(1, 2);

							x_real = ct*t.at<double>(0, 0) + rt*t.at<double>(0, 1) + tx;
							y_real = ct*t.at<double>(1, 0) + rt*t.at<double>(1, 1) + ty;

							x_real3 = c*t.at<double>(0, 0) + r*t.at<double>(0, 1) + centerSLAM.x;
							y_real3 = c*t.at<double>(1, 0) + r*t.at<double>(1, 1) + centerSLAM.y;

							x = (int)round(x_real);//Nearest Neighbourhood interpolation
							y = (int)round(y_real);//Nearest Neighbourhood interpolation

							if (!bilinearInterpolation)
							{
								if (y >= 0 && y < dst_manual.rows)
								{
									if (x >= 0 && x < dst_manual.cols)
									{
										if (Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(r, c) > 0)
										{
											dst_manual.at<float>(y, x) = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(r, c);
											//cout << fixed << std::setprecision(2) << "(" << y << "," << x << ") -->" << Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(r, c) << "  " << endl;
										}
										else
										{
											dst_manual.at<float>(y, x) = 0.5;
										}
									}
								}
							}
							else
							{
								x_real = ct*MI.at<double>(0, 0) + rt*MI.at<double>(0, 1) + centerSLAM.x;
								y_real = ct*MI.at<double>(1, 0) + rt*MI.at<double>(1, 1) + centerSLAM.y;

								float x2 = y_real;
								float y2 = x_real;

								float u = floor(x2);
								float v = floor(y2);

								float W1 = (u + 1 - x2)*(v + 1 - y2);
								float W2 = (x2 - u)*(v + 1 - y2);
								float W3 = (u + 1 - x2)*(y2 - v);
								float W4 = (x2 - u)*(y2 - v);

								if ((u >= 0 && u < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.rows - 1) && (v >= 0 && v < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.cols - 1))
								{
									float res = W1* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v) + W2* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v) + W3* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v + 1) + W4* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v + 1);
									dst_manual.at<float>(r + ty, c + tx) = res;
								}
								else
								{
									dst_manual.at<float>(r + ty, c + tx) = 0.5;
								}
							}
							//t = [cosd(teta) - sind(teta); sind(teta) cosd(teta)]; //matlab code
							//affine = [xscale*cosd(teta) xshear*sind(teta) xtrans; yshear*(-sind(teta)) yscale*cosd(teta) ytrans; 0 0 1]; //matlab code
							//xyz = affine*[(r - origin_x); (c - origin_y); 1]; //matlab code
							//xy = t*[(r - origin_x); (c - origin_y)];

						}
					}
				}

				//cout << "hey";
#endif
#endif

#ifdef WITH_SLAM
				if (Global_occupancyGridParams->useBilinearInterpolation)
					Global_occupancyGridGlobals->occupancyGridMap.set_OccupancyGridSLAMMatrix_ByBilinearInterpolation(*Global_occupancyGridParams);

				//float zr = 60;
				//float xr = 117;


				////		float z = Global_occupancyGridParams->totalRows - (obj.Zw / Global_occupancyGridParams->ZWorldResolution) - 1;
				////		float x = (obj.Xw / Global_occupancyGridParams->XWorldResolution) + Global_occupancyGridParams->middleCol;	

				//float zw = -(zr - Global_occupancyGridParams->totalRows + 1) * Global_occupancyGridParams->ZWorldResolution;
				//float xw = (xr - Global_occupancyGridParams->middleCol) * Global_occupancyGridParams->XWorldResolution;
				//float dr = sqrt(pow(xw, 2) + pow(zw, 2));

				//std::cout << "z:" << zr << " x:" << xr << " zw:" << zw << " xw:" << xw << " dr:" << dr << endl;
/*
#pragma region DrawRealObjects
				//drawObjects on SLAM color
				Mat ColorSLAM;
				cv::cvtColor(Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM, ColorSLAM, CV_GRAY2RGB);

				Mat ColorSLAM_Thresholded;
				cv::cvtColor(Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM_Thresholded, ColorSLAM_Thresholded, CV_GRAY2RGB);

				Mat ColorOcc;
				cv::cvtColor(Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix, ColorOcc, CV_GRAY2RGB);

				int Global_currentIMGNum = Global_minIMGIndex + Global_currentIMGIndex * Global_imgIncrement;
				int mult = 2;

				if (Global_currentIMGNum == 19)
					addDrawPoint(19, "s1", 105, 104, 13, 5, ColorOcc, ColorSLAM, 4.3);


				if (Global_currentIMGNum == 29)
				{
					addDrawPoint(29, "re", 100, 114, 12, 4, ColorOcc, ColorSLAM, 4.97);
					addDrawPoint(29, "c1", 115, 98, 5, 3, ColorOcc, ColorSLAM, 3.34);
				}

				//if (Global_currentIMGNum == 39)
				//{
				//	addDrawPoint(39, "d1", 59, 114, 7, 4, ColorOcc, ColorSLAM, 8.8);
				//}

				if (Global_currentIMGNum == 49)
				{
					addDrawPoint(49, "y1", 92, 123, 6, 3, ColorOcc, ColorSLAM, 6.14);
					addDrawPoint(49, "d1", 58, 81, 41, 5, ColorOcc, ColorSLAM, 8.8);
				}

				//if (Global_currentIMGNum == 79)
				//{
				//	addDrawPoint(79, "sl2", 112, 105, 6, 3, ColorOcc, ColorSLAM, 3.77);
				//	addDrawPoint(79, "sr2", 115, 113, 6, 3, ColorOcc, ColorSLAM, 3.77);
				//}


				if (Global_currentIMGNum == 89)
				{
					addDrawPoint(89, "s2", 113, 87, 13, 3, ColorOcc, ColorSLAM, 3.77);
				}

				//if (Global_currentIMGNum == 109)
				//{
				//	addDrawPoint(86, "s2", 92, 122, 7, 3, ColorOcc, ColorSLAM, 3.77);
				//}

				if (Global_currentIMGNum == 119)
				{
					addDrawPoint(86, "c2", 88, 100, 6, 4, ColorOcc, ColorSLAM, 6.05);
				}

				if (Global_currentIMGNum == 129)
				{
					addDrawPoint(86, "d2", 51, 72, 27, 6, ColorOcc, ColorSLAM, 9.2);
				}

				if (Global_currentIMGNum == 139)
				{
					addDrawPoint(86, "sa", 99, 110, 8, 3, ColorOcc, ColorSLAM, 4.95);
					addDrawPoint(86, "s3", 63, 111, 10, 7, ColorOcc, ColorSLAM, 8.35);
				}

				for (int i = 0; i < drawings.size(); i++)
				{
					cv::rectangle(ColorSLAM, drawings[i]->p1, drawings[i]->p2, cv::Scalar(0, 255, 0), 1 * mult);
					cv::rectangle(ColorSLAM_Thresholded, drawings[i]->p1, drawings[i]->p2, cv::Scalar(0, 255, 0), 1 * mult);

					ostringstream dt;
					dt << fixed << std::setprecision(2) << " - " << drawings[i]->distance << " (%" << drawings[i]->Pou << ")";


					if (i % 4 == 0)
					{
						Point2i textPoint(drawings[i]->p1.x + 5 * mult, drawings[i]->p1.y);
						cv::putText(ColorSLAM, drawings[i]->text, drawings[i]->p1, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(0, 0, 255), 0.35*mult, CV_AA);
						cv::putText(ColorSLAM, dt.str().c_str(), textPoint, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(0, 0, 255), 0.35*mult, CV_AA);

						cv::putText(ColorSLAM_Thresholded, drawings[i]->text, drawings[i]->p1, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(0, 0, 255), 0.35*mult, CV_AA);
						cv::putText(ColorSLAM_Thresholded, dt.str().c_str(), textPoint, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(0, 0, 255), 0.35*mult, CV_AA);
					}
					else
					{
						if (i % 4 == 1)
						{
							cv::putText(ColorSLAM, drawings[i]->text, drawings[i]->p2, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(255, 0, 0), 0.35*mult, CV_AA);
							cv::putText(ColorSLAM_Thresholded, drawings[i]->text, drawings[i]->p2, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(255, 0, 0), 0.35*mult, CV_AA);
							Point2i textPoint(drawings[i]->p2.x + 5 * mult, drawings[i]->p2.y);
							cv::putText(ColorSLAM, dt.str().c_str(), textPoint, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(255, 0, 0), 0.35*mult, CV_AA);
							cv::putText(ColorSLAM_Thresholded, dt.str().c_str(), textPoint, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(255, 0, 0), 0.35*mult, CV_AA);
						}
						else
						{
							if (i % 4 == 2)
							{
								cv::putText(ColorSLAM, drawings[i]->text, drawings[i]->p1, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(0, 255, 0), 0.35*mult, CV_AA);
								cv::putText(ColorSLAM_Thresholded, drawings[i]->text, drawings[i]->p1, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(0, 255, 0), 0.35*mult, CV_AA);
								Point2i textPoint(drawings[i]->p1.x + 5 * mult, drawings[i]->p1.y);
								cv::putText(ColorSLAM, dt.str().c_str(), textPoint, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(0, 255, 0), 0.35*mult, CV_AA);
								cv::putText(ColorSLAM_Thresholded, dt.str().c_str(), textPoint, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(0, 255, 0), 0.35*mult, CV_AA);
							}
							else
							{
								cv::putText(ColorSLAM, drawings[i]->text, drawings[i]->p2, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(255, 0, 255), 0.35*mult, CV_AA);
								cv::putText(ColorSLAM_Thresholded, drawings[i]->text, drawings[i]->p2, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(255, 0, 255), 0.35*mult, CV_AA);
								Point2i textPoint(drawings[i]->p2.x + 5 * mult, drawings[i]->p2.y);
								cv::putText(ColorSLAM, dt.str().c_str(), textPoint, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(255, 0, 255), 0.35*mult, CV_AA);
								cv::putText(ColorSLAM_Thresholded, dt.str().c_str(), textPoint, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(255, 0, 255), 0.35*mult, CV_AA);
							}

						}

					}

					Point2i closest_P1(drawings[i]->closes_Point.x - 1, drawings[i]->closes_Point.y - 1);
					Point2i closest_P2(drawings[i]->closes_Point.x + 1, drawings[i]->closes_Point.y + 1);

					cv::rectangle(ColorSLAM, closest_P1, closest_P2, cv::Scalar(0, 255, 255), 1 * mult);
					cv::rectangle(ColorSLAM_Thresholded, closest_P1, closest_P2, cv::Scalar(0, 255, 255), 1 * mult);

				}
#pragma endregion DrawRealObjects
				*/
				//cout << "";

#endif
				//for (size_t obj_index = 0; obj_index < Global_mergedObjectList2.size(); obj_index++)
				//{
				//	MyObject obj = Global_mergedObjectList2[obj_index];
				//	if ((!obj.isDeleted) && (obj.Zw < Global_occupancyGridParams->ZWorldLimit) && (abs(obj.Xw) < Global_occupancyGridParams->XWorldLimit) && (obj.Yw > 0.5))
				//	{
				//		int objLeftd, objLeftv, objRightd, objRightv, objLeftu, objRightu;
				//		float objLeft_Xw, objLeft_Yw, objLeft_Zw, objRight_Xw, objRight_Yw, objRight_Zw;

				//		if (obj.isLeftLeaning)
				//		{
				//			objLeftu = obj.xLow_Index;
				//			objRightu = obj.xHigh_Index;
				//			objLeftd = obj.yhighDisp_Index;
				//			objLeftv = obj.yhighDisp_Vmin_Index;
				//			objRightd = obj.ylowDisp_Index;
				//			objRightv = obj.ylowDisp_Vmin_Index;
				//		}
				//		else
				//		{
				//			objLeftu = obj.xLow_Index;
				//			objRightu = obj.xHigh_Index;
				//			objLeftd = obj.ylowDisp_Index;
				//			objLeftv = obj.ylowDisp_Vmin_Index;
				//			objRightd = obj.yhighDisp_Index;
				//			objRightv = obj.yhighDisp_Vmin_Index;
				//		}
				//								
				//		ImageToWorldCoordinates(objLeftu, objLeftv, objLeftd, objLeft_Xw, objLeft_Yw, objLeft_Zw);
				//		ImageToWorldCoordinates(objRightu, objRightv, objRightd, objRight_Xw, objRight_Yw, objRight_Zw);

				//		float z_Left = Global_occupancyGridParams->totalRows - (objLeft_Zw / Global_occupancyGridParams->ZWorldResolution) - 1;
				//		float x_Left = (objLeft_Xw / Global_occupancyGridParams->XWorldResolution) + Global_occupancyGridParams->middleCol;
				//		cv::Point2i oldPoint_Left((int)x_Left, (int)z_Left);						
				//		cv::Point2i newPoint_left = Global_occupancyGridParams->setMapCoordinatesByNN(oldPoint_Left.x, oldPoint_Left.y);

				//		float z_Right = Global_occupancyGridParams->totalRows - (objRight_Zw / Global_occupancyGridParams->ZWorldResolution) - 1;
				//		float x_Right = (objRight_Xw / Global_occupancyGridParams->XWorldResolution) + Global_occupancyGridParams->middleCol;
				//		cv::Point2i oldPoint_Right((int)x_Right, (int)z_Right);
				//		cv::Point2i newPoint_Right = Global_occupancyGridParams->setMapCoordinatesByNN(oldPoint_Right.x, oldPoint_Right.y);

				//		cv::line(ColorOcc, oldPoint_Left, oldPoint_Right, cv::Scalar(255, 0, 255), 1);
				//		cv::line(ColorSLAM, newPoint_left, newPoint_Right, cv::Scalar(255, 0, 255), 1);

				//		float z = Global_occupancyGridParams->totalRows - (obj.Zw / Global_occupancyGridParams->ZWorldResolution) - 1;
				//		float x = (obj.Xw / Global_occupancyGridParams->XWorldResolution) + Global_occupancyGridParams->middleCol;						

				//		cv::Point2i oldPoint((int)x, (int)z);
				//		cv::circle(ColorOcc, oldPoint, 3, cv::Scalar(255, 0, 0));
				//		cv::Point2i new_xy = Global_occupancyGridParams->setMapCoordinatesByNN(x, z);
				//		cv::circle(ColorSLAM, new_xy, 3, cv::Scalar(255, 0, 0));

				//		std::cout << "index:" << obj_index << " hi_d:" << obj.yhighDisp_Index << " low_d:" << obj.ylowDisp_Index << " hir_d:" << obj.yhighDisp_Index_real << " lowr_d:" << obj.ylowDisp_Index_real   << " dd:" << obj.directDistance << " rd:" << obj.realDistanceTiltCorrected << " Zw:" << obj.Zw << " Xw:" << obj.Xw << " Yw:" << obj.Yw << " W:" << obj.objWidth << " oX:" << (int)x << " oZ:" << (int)z << " sX:" << new_xy.x << " sZ:" << new_xy.y<< std::endl;
				//		std::cout << "index:" << obj_index << " ll:" << obj.isLeftLeaning << " lu:" << objLeftu << " lv:" << objLeftv  << " ld:" << objLeftd << " ru:" << objRightu << " rv:" << objRightv << " rd:" << objRightd << " xl:" << (int)x_Left << " zl:" << (int)z_Left << " xr:" << (int)x_Right << " zr:" << (int)z_Right << endl << endl;
				//	}
				//}

#ifdef WITH_DOCK_DETECTION
				Mat line_draws;
				getBoundingBoxes(Global_Current_Left_Image, dispD, drawed, Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix, line_draws);

#endif

				Global_uDisparityGlobals->clear();
				Global_vDisparityGlobals->clear();
				Global_dDisparityGlobals->clear();

#ifdef WITH_OCCUPANCY_GRID
				Mat check_occupancy = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix;
#endif

#ifdef WITH_SLAM
				Mat check_occupancy_SLAM = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM;
				Mat check_occupancy_SLAM_Thresholded = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM_Thresholded;
#else
#ifdef WITH_OCCUPANCY_GRID
				Mat check_occupancy_thresholded = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix_Thresholded;
#endif
#endif

				if (showImages)
				{
					//cv::imshow("Window1", Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix);
					//cv::imshow("Window1", Global_uDisparityGlobals->uDisparityImage);
					cv::imshow("Window1", uDispDraw);

					cv::imshow("Window4", Global_Current_DenseDisparity_Image);
					//cv::imshow("Window6", colorV);


					//cv::imshow("Window1", Global_uDisparityGlobals->uDisparityImage);
					//cv::imshow("Window2", Global_uDisparityGlobals->uDisparityImage);
					//cv::imshow("Window3", line_draws);
					//cv::imshow("Window4", colorV);
					//cv::imshow("Window5", Global_uDisparityGlobals->uDisparityImage_thresholded);
#ifdef WITH_OCCUPANCY_GRID
					//cv::imshow("Window6", Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix);
					cv::imshow("Window3", Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix);

#ifdef WITH_SLAM
					//cv::imshow("Window7", Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM);
					cv::imshow("Window2", Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM_Thresholded);
#else
					cv::imshow("Window7", drawed);
#endif

#endif
#ifdef WITH_LANE_DETECTION
					Mat cannyResized;
					int width = (detLanes->dstCanny.cols + 1) * 0.75;
					int height = (detLanes->dstCanny.rows + 1) * 0.75;
					cv::resize(detLanes->dstCanny, cannyResized, cv::Size(width, height));
					cv::imshow("Window6", Global_Current_DenseDisparity_Image);
					cv::imshow("Window7", cannyResized);
#endif
					//cv::imshow("Window8", Global_Current_DenseDisparity_Image);

					int c = (cvWaitKey(1000) & 0xff);
					if (c == 32)
					{
						c = (cvWaitKey(1000) & 0xff);
						while (c != 32)
						{
							c = (cvWaitKey(3000) & 0xff);
						}
					}//cvWaitKey if
				}//if show images
			}//Offline Imagelist For loop
		}//if not using GPU
		else//if use GPU
		{
#ifdef WITH_GPU
			for (int imgindex = 0; imgindex < Global_Buffer_Size; imgindex++)
			{
				Global_currentIMGIndex = imgindex;

				Global_Current_Left_Image = Global_IMG_Left_Image_Buffer_GPU[imgindex];
				Global_Current_Right_Image = Global_IMG_Right_Image_Buffer_GPU[imgindex];

				setDenseDisparityMapCUDA();
				set_UVDisparityMaps();
				set_BestLine_From_VDisparity();
				set_ImageInfinityPoint_And_VehicleRoadLine(Global_imgMiddleCol, Global_vDisparityGlobals->bestRoadLine_vIntersection, Global_imgTotalCols, Global_imgTotalRows);

				set_LabelsOf_UDisparityLines_Using_LabelingAlg();
				//cv::Mat draw1 = drawUDispObjects(uDispDraw1, Global_labeledObjectList);
				merge_MaskVObjects_From_UDisparity_By_Labeling_Alg(Global_labeledObjectList);
				//cv::Mat draw2 = drawUDispObjects(uDispDraw2, Global_mergedObjectList1);
				merge_InclinedObjects_From_UDisparity_By_Labeling_Alg(Global_mergedObjectList1);
				//cv::Mat draw3 = drawUDispObjects(uDispDraw3, Global_mergedObjectList2);
				set_SideBuildingsIfAny_And_Filter(Global_mergedObjectList2);
				createDenseDispObjects(Global_mergedObjectList2);

#ifdef WITH_LANE_DETECTION
				laneDetection *detLanes = new laneDetection(1.05, Global_Current_Left_Image);
				detLanes->testLanes(true, Global_Current_Left_Image);
				cv::imshow("Window1", Global_Current_Left_Image);
				int c = (cvWaitKey(1000) & 0xff);
				Mat laneIMG = detLanes->getLanes(false, Global_vDisparityGlobals->bestRoadLine_vIntersection);
#endif	

				//cv::Mat draw4 = drawUDispObjects(uDispDraw4, Global_mergedObjectList2);		
				cv::Mat colorV;
				cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV, CV_GRAY2BGR);
				cv::line(colorV, Point(0, Global_vDisparityGlobals->bestRoadLine_vIntersection), Point(Global_vDisparityGlobals->bestRoadLine_dInterSection, Global_vDisparityParams->totalRows - 1), cv::Scalar(0, 0, 255));
				cv::line(colorV, Point(Global_vDisparityGlobals->piecewiseBestLine[0], Global_vDisparityGlobals->piecewiseBestLine[1]), Point(Global_vDisparityGlobals->piecewiseBestLine[2], Global_vDisparityGlobals->piecewiseBestLine[3]), cv::Scalar(0, 255, 0));

				cv::Mat drawed = drawObjects(Global_Current_Left_Image, Global_mergedObjectList2);
				cv::Mat uDispDraw = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, drawed.type());
				uDispDraw = drawUDispObjects(uDispDraw, Global_mergedObjectList2);

				int deleted = getNumObjectsDeleted(Global_mergedObjectList2);
				int livingObj = Global_currentNumOfObjects - deleted;
				//diff_disparity = (static_cast<float>(t2)-static_cast<float>(t1));
				//cout << "NumOfObjs:" << Global_currentNumOfObjects << " LivingObjects:" << livingObj << " time: " << diff_disparity << "ms." << endl;

#ifdef WITH_OCCUPANCY_GRID				
				set_occupancyUDisparityMap();
#endif

				Global_uDisparityGlobals->clear();
				Global_vDisparityGlobals->clear();
				Global_dDisparityGlobals->clear();

				if (showImages)
				{
					cv::imshow("Window1", Global_uDisparityGlobals->uDisparityImage);
					cv::imshow("Window2", uDispDraw);
					cv::imshow("Window3", drawed);
					cv::imshow("Window4", colorV);
					cv::imshow("Window5", Global_uDisparityGlobals->uDisparityImage_thresholded);
#ifdef WITH_OCCUPANCY_GRID
					cv::imshow("Window6", Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix);
#endif
#ifdef WITH_LANE_DETECTION
					Mat cannyResized;
					int width = (detLanes->dstCanny.cols + 1) * 0.75;
					int height = (detLanes->dstCanny.rows + 1) * 0.75;
					cv::resize(detLanes->dstCanny, cannyResized, cv::Size(width, height));
					cv::imshow("Window6", Global_Current_DenseDisparity_Image);
					cv::imshow("Window7", cannyResized);
#endif
					cv::imshow("Window8", Global_Current_DenseDisparity_Image);

					int c = (cvWaitKey(1000) & 0xff);
					if (c == 32)
					{
						c = (cvWaitKey(1000) & 0xff);
						while (c != 32)
						{
							c = (cvWaitKey(3000) & 0xff);
						}
					}//cvWaitKey if
				}//if show images
			}
#else
			cout << "You should Define GPU...(#define WITH_GPU)" << endl;
#endif
		}//else use GPU
#pragma endregion Offline		
	}
#pragma region Online
	else//use realtime
	{
		TriclopsError triclops_status;
		TriclopsContext context = nullptr;
		TriclopsMonoStereoPair monoStereoPair;

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

		// camera to be used to grab color image
		FC2::Camera camera;
		// grabbed unprocessed image
		FC2::Image grabbedImage;

		// connect camera
		FC2::Error flycap_status = camera.Connect();

		FC2T::ErrorType fc2TriclopsError;

		FC2::CameraInfo camInfo;
		FC2::Error fc2Error = camera.GetCameraInfo(&camInfo);
		fc2TriclopsError = FC2T::getContextFromCamera(camInfo.serialNumber, &context);

		fc2Error = camera.StartCapture();

		const char* errC = fc2Error.GetDescription();
		std::string err = errC;

		if (!Global_useGPU)//use CPU buffer
		{
			while (strcmp(err.c_str(), "Ok.") == 0)
			{
				fc2Error = camera.RetrieveBuffer(&grabbedImage);
				errC = fc2Error.GetDescription();
				err = errC;

				ImageContainer imageCont;
				generatemonoStereoPair(context, grabbedImage, imageCont, monoStereoPair);

				// Prepare context for frame rectification
				triclops_status = triclopsPrepareRectificationData(context, bumblebeeObject->outputHeight, bumblebeeObject->outputWidth, monoStereoPair.right.nrows, monoStereoPair.right.ncols);//could be beneficial in multi CPU code or paralel computing
																																										 //Rectify
				triclops_status = triclopsRectify(context, &monoStereoPair);
				//triclops_examples::handleError("triclopsRectify()", triclops_status, __LINE__);

				TriclopsImage imageL, imageR;
				triclops_status = triclopsGetImage(context, TriImg_RECTIFIED, TriCam_RIGHT, &imageR);
				triclops_status = triclopsGetImage(context, TriImg_RECTIFIED, TriCam_LEFT, &imageL);
				//triclops_examples::handleError("triclopsGetImage()", triclops_status, __LINE__);

				Global_currentIMGIndex = 0;
				Global_Current_Left_Image = convertTriclops2OpencvMat(imageL);
				Global_Current_Right_Image = convertTriclops2OpencvMat(imageR);

				set_DenseDisparityMap();
				set_UVDisparityMaps();
				//set_BestLine_From_VDisparity();
				//set_BestLine_From_VDisparity_Curve();
				set_BestLine_From_VDisparity_Regression();
				//set_ImageInfinityPoint_And_VehicleRoadLine(Global_imgMiddleCol, Global_vDisparityGlobals->bestRoadLine_vIntersection, Global_imgTotalCols, Global_imgTotalRows);
				set_ImageInfinityPoint_And_VehicleRoadLine(Global_imgMiddleCol, Global_vDisparityGlobals->piecewiseBestLine_RoadEndPoint.y, Global_imgTotalCols, Global_imgTotalRows);

				//cv::Mat uDispDraw1 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, CV_8UC3);
				//cv::Mat uDispDraw2 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, CV_8UC3);
				//cv::Mat uDispDraw3 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, CV_8UC3);
				//cv::Mat uDispDraw4 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, CV_8UC3);
				//cv::Mat uDispDraw5 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, CV_8UC3);

				//diff_disparity = (static_cast<float>(t2)-static_cast<float>(t1));
				//cout << "Time: " << diff_disparity << " ms." << endl;

				//Mat draw0(Global_uDisparityGlobals->uDisparityImage);

				set_LabelsOf_UDisparityLines_Using_LabelingAlg();

				//Mat draw01(Global_uDisparityGlobals->uDisparityImage_thresholded);

				//cv::Mat draw1 = drawUDispObjects(uDispDraw1, Global_labeledObjectList);
				merge_MaskVObjects_From_UDisparity_By_Labeling_Alg(Global_labeledObjectList);
				//cv::Mat draw2 = drawUDispObjects(uDispDraw2, Global_mergedObjectList1);
				merge_InclinedObjects_From_UDisparity_By_Labeling_Alg(Global_mergedObjectList1);
				//cv::Mat draw3 = drawUDispObjects(uDispDraw3, Global_mergedObjectList2);
				set_SideBuildingsIfAny_And_Filter(Global_mergedObjectList2);
				//cv::Mat draw4 = drawUDispObjects(uDispDraw4, Global_mergedObjectList2);
				createDenseDispObjects(Global_mergedObjectList2);
				//cv::Mat draw5= drawUDispObjects(uDispDraw5, Global_mergedObjectList2);

#ifdef WITH_LANE_DETECTION
				laneDetection *detLanes = new laneDetection(1.05, Global_Current_Left_Image);
				detLanes->testLanes(true, Global_Current_Left_Image);
				cv::imshow("Window1", Global_Current_Left_Image);
				int c = (cvWaitKey(1000) & 0xff);
				Mat laneIMG = detLanes->getLanes(false, Global_vDisparityGlobals->bestRoadLine_vIntersection);
#endif			


				//cv::Mat draw4 = drawUDispObjects(uDispDraw4, Global_mergedObjectList2);				

				cv::Mat colorV;
				cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV, CV_GRAY2BGR);
				//cv::line(colorV, Point(0, Global_vDisparityGlobals->bestRoadLine_vIntersection), Point(Global_vDisparityGlobals->bestRoadLine_dInterSection, Global_vDisparityParams->totalRows - 1), cv::Scalar(255, 0, 255));

				if (Global_vDisparityGlobals->bestRoadLine_a == 0)
				{
					cv::line(colorV, Point(Global_vDisparityGlobals->bestRoadLine[0], Global_vDisparityGlobals->bestRoadLine[1]), Point(Global_vDisparityGlobals->bestRoadLine[2], Global_vDisparityGlobals->bestRoadLine[3]), cv::Scalar(0, 0, 255));
					cv::line(colorV, Point(Global_vDisparityGlobals->piecewiseBestLine[0], Global_vDisparityGlobals->piecewiseBestLine[1]), Point(Global_vDisparityGlobals->piecewiseBestLine[2], Global_vDisparityGlobals->piecewiseBestLine[3]), cv::Scalar(0, 255, 0));
					circle(colorV, Global_vDisparityGlobals->piecewiseBestLine_RoadEndPoint, 3, Scalar(255, 255, 0), 2);
					circle(colorV, Global_vDisparityGlobals->piecewiseBestLine_RoadStartPoint, 3, Scalar(255, 255, 0), 2);
				}
				else
				{
					std::vector<Point2f> curvePoints;

					int start_point_x = 0;
					int end_point_x = Global_vDisparityParams->totalCols - 1;
					for (float x = start_point_x; x <= end_point_x; x += 1) {
						float y = Global_vDisparityGlobals->bestRoadLine_m*x + Global_vDisparityGlobals->bestRoadLine_a*x*x + Global_vDisparityGlobals->bestRoadLine_b;
						Point2f new_point = Point2f(x, y);                  //resized to better visualize
						curvePoints.push_back(new_point);
					}

					Mat curve(curvePoints, true);
					curve.convertTo(curve, CV_32S); //adapt type for polylines

					cv::line(colorV, Point(Global_vDisparityGlobals->piecewiseBestLine[0], Global_vDisparityGlobals->piecewiseBestLine[1]), Point(Global_vDisparityGlobals->piecewiseBestLine[2], Global_vDisparityGlobals->piecewiseBestLine[3]), cv::Scalar(0, 255, 0));
					polylines(colorV, curve, false, Scalar(255), 2, CV_AA);
					circle(colorV, Global_vDisparityGlobals->piecewiseBestLine_RoadEndPoint, 3, Scalar(255, 255, 0), 2);
					circle(colorV, Global_vDisparityGlobals->piecewiseBestLine_RoadStartPoint, 3, Scalar(255, 255, 0), 2);
				}

				//int v = get_v_From_VDisparity_RoadProfile(24);
				cv::Mat dispV = Global_vDisparityGlobals->vDisparityImage;
				cv::Mat dispU = Global_uDisparityGlobals->uDisparityImage;
				cv::Mat dispUT = Global_uDisparityGlobals->uDisparityImage_thresholded;
				cv::Mat dispD = Global_Current_DenseDisparity_Image;
				//cv::Mat drawed = drawObjects(Global_Current_Left_Image, Global_mergedObjectList2);

				cv::Mat drawed;
				cv::cvtColor(Global_Current_Left_Image, drawed, CV_GRAY2RGB);

				cv::Mat uDispDraw = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, drawed.type());
				uDispDraw = drawUDispObjects(uDispDraw, Global_mergedObjectList2);

				//int deleted = getNumObjectsDeleted(Global_mergedObjectList2);
				//int livingObj = Global_currentNumOfObjects - deleted;
				//diff_disparity = (static_cast<float>(t2)-static_cast<float>(t1));
				//cout << "NumOfObjs:" << Global_currentNumOfObjects << " LivingObjects:" << livingObj << " time: " << diff_disparity << "ms." << endl;

#ifdef WITH_OCCUPANCY_GRID				
				set_occupancyUDisparityMap();
#ifdef WITH_SLAM
				//if (Global_occupancyGridParams->useJointProbilities)//delete later, just for testing
				//{
				//	Mat check_PCu, check_PVu, check_POu;
				//	check_POu = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix;
				//	check_PCu = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixPCu;
				//	check_PVu = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixPVu;
				//	cout << "";
				//}
#endif

				//Global_occupancyGridGlobals->occupancyGridMap.
				//Global_occupancyGridGlobals->occupancyGridMap.print_OccupancyGridMatrix();
#ifdef WITH_SLAM_TEST
				//centerSLAM.x = Global_occupancyGridParams->middleCol;// round((Global_occupancyGridGlobals->occupancyGridMap.middleColSLAM + Global_occupancyGridGlobals->occupancyGridMap.middleRowSLAM + Global_occupancyGridGlobals->occupancyGridMap.middleCol + Global_occupancyGridGlobals->occupancyGridMap.middleRow) / 2.0);// Global_occupancyGridGlobals->occupancyGridMap.middleCol; //600; //(double)Global_occupancyGridGlobals->occupancyGridMap.middleCol;
				//centerSLAM.y = Global_occupancyGridParams->totalRows;// round(((Global_occupancyGridGlobals->occupancyGridMap.middleRowSLAM + Global_occupancyGridGlobals->occupancyGridMap.middleRow) - (Global_occupancyGridGlobals->occupancyGridMap.middleColSLAM - Global_occupancyGridGlobals->occupancyGridMap.middleCol)) / 2.0); //Global_occupancyGridGlobals->occupancyGridMap.middleColSLAM;
				//centerSLAM.x = Global_occupancyGridGlobals->occupancyGridMap.middleColSLAM;
				//centerSLAM.y = Global_occupancyGridGlobals->occupancyGridMap.totalRowsSL;

				/*
				int noOfObservations = 11;
				float NpUD = 100;
				float* NvUD = new float[noOfObservations] { 100, 90, 50, 50, 0, 100, 80, 70, 50, 100, 100};
				float* NoUD = new float[noOfObservations] { 100, 90, 50, 40, 0, 70, 60, 50, 10, 90, 100};
				float PVu, RoUD, PCu, POu, POu2;
				float aprioProbility = 0.5, aprioProbility2 = 1.0;
				Global_occupancyGridParams->uDisp_probabilityFP = 0.0;
				Global_occupancyGridParams->uDisp_probabilityFN = 0.0;

				float PA, PB, PAB, PBA;

				for (int expNum = 0; expNum < noOfObservations; expNum++)
				{
				PVu = NvUD[expNum] / NpUD;

				if (NvUD[expNum] > 0)
				RoUD = NoUD[expNum] / NvUD[expNum];
				else
				RoUD = 0;

				//PCu = 1 - exp(-(RoUD / Global_occupancyGridParams->uDisp_tO));
				PCu = RoUD;

				POu = PVu*PCu*(1 - Global_occupancyGridParams->uDisp_probabilityFP) + PVu*(1 - PCu)*Global_occupancyGridParams->uDisp_probabilityFN + (1 - PVu)*aprioProbility;
				POu2 = PVu*PCu*(1 - Global_occupancyGridParams->uDisp_probabilityFP) + PVu*(1 - PCu)*Global_occupancyGridParams->uDisp_probabilityFN + (1 - PVu)*aprioProbility2;
				float orgPOu = PVu*PCu*(1 - Global_occupancyGridParams->uDisp_probabilityFP) + PVu*(1 - PCu)*Global_occupancyGridParams->uDisp_probabilityFN + (1 - PVu)*0.5;

				cout << fixed << std::setprecision(2) << "AprioP:" << aprioProbility << " FPR:" << Global_occupancyGridParams->uDisp_probabilityFP << " FNR:" << Global_occupancyGridParams->uDisp_probabilityFN << " Np:" << NpUD << " Nv:" << NvUD[expNum] << " No:" << NoUD[expNum] << " --> PVu:" << PVu << " Ro:" << RoUD << " PCu:" << PCu << " POu:" << POu <<  " orgPOu:" << orgPOu << endl;

				PA = aprioProbility2;
				PBA = POu2;
				PB = orgPOu;

				PAB = (PBA*PA) / PB;

				cout << fixed << std::setprecision(2) << "PA (AprioP):" << PA << " PBA (new POU2):" << PBA << " PB (org PB):" << PB << " PAB (Result):" << PAB << endl << endl;

				aprioProbility = POu;
				aprioProbility2 = PBA;
				}
				*/
				Point2f centerSLAM;
				//centerSLAM.x = Global_occupancyGridParams->middleColSLAM - Global_occupancyGridParams->middleCol;
				//centerSLAM.y = Global_occupancyGridParams->middleRowSLAM - Global_occupancyGridParams->totalRows;

				centerSLAM.x = Global_occupancyGridParams->middleCol;
				centerSLAM.y = Global_occupancyGridParams->totalRows;
				float tx = Global_occupancyGridParams->middleColSLAM;
				float ty = Global_occupancyGridParams->middleRowSLAM;

				//float tx = round(Global_occupancyGridParams->middleColSLAM - centerSLAM.x);
				//float ty = round(Global_occupancyGridParams->middleRowSLAM - centerSLAM.y);

				double rotateDegree = 0;

				//print rotate matrix
				Mat M = getRotationMatrix2D(centerSLAM, rotateDegree, 1);
				Mat MI;
				invertAffineTransform(M, MI);


				Mat t = cv::Mat::zeros(2, 3, CV_64FC1); //clockwise //To represent affine transformations with matrices, we can use homogeneous coordinates.This means representing a 2 - vector(x, y) as a 3 - vector(x, y, 1),

				t.at<double>(0, 0) = cos(rotateDegree * CV_PI / 180.0);//cos(teta)
				t.at<double>(0, 1) = sin(rotateDegree * CV_PI / 180.0);//sin(teta)
				t.at<double>(0, 2) = centerSLAM.y;//y trans
				t.at<double>(1, 0) = -sin(rotateDegree * CV_PI / 180.0);//-sin(teta)
				t.at<double>(1, 1) = cos(rotateDegree * CV_PI / 180.0);//cosd(teta)
				t.at<double>(1, 2) = centerSLAM.x;//x trans

												  //cout << "M" << endl;

												  //for (int r = 0; r < M.rows; r++)
												  //{
												  //	for (int c = 0; c < M.cols; c++)
												  //	{
												  //		cout << fixed << std::setprecision(2) << M.at<double>(r,c) << "  ";
												  //	}
												  //	cout << endl;
												  //}

												  //cout << "MI" << endl;

												  //for (int r = 0; r < M.rows; r++)
												  //{
												  //	for (int c = 0; c < M.cols; c++)
												  //	{
												  //		cout << fixed << std::setprecision(2) << MI.at<double>(r, c) << "  ";
												  //	}
												  //	cout << endl;
												  //}

												  //cout << "M_Manual" << endl;

												  //for (int r = 0; r < M.rows; r++)
												  //{
												  //	for (int c = 0; c < M.cols; c++)
												  //	{
												  //		cout << fixed << std::setprecision(2) << t.at<double>(r, c) << "  ";
												  //	}
												  //	cout << endl;
												  //}


				Mat dst0 = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix;
				Mat dst1 = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM;
				//Mat dst_manual = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM;
				Mat dst_manual = Mat::zeros(Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM.rows, Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM.cols, CV_32FC1);

				Mat dst;
				warpAffine(Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix, dst, M, Size(Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM.cols, Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM.rows));
				//warpAffine(Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix, dst, M, Size(Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.cols, Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.rows));


				bool bilinearInterpolation = true;
				bool bilinearInterpolationExtended = true;

				float x_real, y_real, x_real2, y_real2, x_real3, y_real3;
				int x, y;
				float ct, rt;

				//find boundaries not to search all pixels to interpolate
				float *xPoints = new float[4];
				float *yPoints = new float[4];

				//[0,0]
				float rectLeftUpPointX = 0, rectLeftUpPointX_map;
				float rectLeftUpPointY = 0, rectLeftUpPointY_map;
				ct = (rectLeftUpPointX - centerSLAM.x);// (c - origin_x)
				rt = (rectLeftUpPointY - centerSLAM.y);//(r - origin_y)
				rectLeftUpPointX_map = ct*M.at<double>(0, 0) + rt*M.at<double>(0, 1) + tx;
				rectLeftUpPointY_map = ct*M.at<double>(1, 0) + rt*M.at<double>(1, 1) + ty;
				xPoints[0] = rectLeftUpPointX_map;
				yPoints[0] = rectLeftUpPointY_map;

				//[0,c]
				float rectRightUpPointX = Global_occupancyGridParams->totalCols, rectRightUpPointX_map;
				float rectRightUpPointY = 0, rectRightUpPointY_map;
				ct = (rectRightUpPointX - centerSLAM.x);// (c - origin_x)
				rt = (rectRightUpPointY - centerSLAM.y);//(r - origin_y)
				rectRightUpPointX_map = ct*M.at<double>(0, 0) + rt*M.at<double>(0, 1) + tx;
				rectRightUpPointY_map = ct*M.at<double>(1, 0) + rt*M.at<double>(1, 1) + ty;
				xPoints[1] = rectRightUpPointX_map;
				yPoints[1] = rectRightUpPointY_map;

				//[r,0]
				float rectLeftDownPointX = 0, rectLeftDownPointX_map;
				float rectLeftDownPointY = Global_occupancyGridParams->totalRows, rectLeftDownPointY_map;
				ct = (rectLeftDownPointX - centerSLAM.x);// (c - origin_x)
				rt = (rectLeftDownPointY - centerSLAM.y);//(r - origin_y)
				rectLeftDownPointX_map = ct*M.at<double>(0, 0) + rt*M.at<double>(0, 1) + tx;
				rectLeftDownPointY_map = ct*M.at<double>(1, 0) + rt*M.at<double>(1, 1) + ty;
				xPoints[2] = rectLeftDownPointX_map;
				yPoints[2] = rectLeftDownPointY_map;

				//[r,c]
				float rectRightDownPointX = Global_occupancyGridParams->totalCols, rectRightDownPointX_map;
				float rectRightDownPointY = Global_occupancyGridParams->totalRows, rectRightDownPointY_map;
				ct = (rectRightDownPointX - centerSLAM.x);// (c - origin_x)
				rt = (rectRightDownPointY - centerSLAM.y);//(r - origin_y)
				rectRightDownPointX_map = ct*M.at<double>(0, 0) + rt*M.at<double>(0, 1) + tx;
				rectRightDownPointY_map = ct*M.at<double>(1, 0) + rt*M.at<double>(1, 1) + ty;
				xPoints[3] = rectRightDownPointX_map;
				yPoints[3] = rectRightDownPointY_map;

				//find the smallest and biggest x and y points to draw rectangle boundaries to search for interpolation

				//for rows (x Points)
				float rStart = Global_occupancyGridParams->totalRowsSLAM;
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

				if (rEnd > Global_occupancyGridParams->totalRowsSLAM)
					rEnd = Global_occupancyGridParams->totalRowsSLAM;

				//for columns (y Points)
				float cStart = Global_occupancyGridParams->totalColsSLAM;
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

				if (cEnd > Global_occupancyGridParams->totalColsSLAM)
					cEnd = Global_occupancyGridParams->totalColsSLAM;

				int rStarti = (int)round(rStart);
				int cStarti = (int)round(cStart);
				int rEndi = (int)round(rEnd);
				int cEndi = (int)round(cEnd);
				//find boundaries not to search all pixels to interpolate

				if (bilinearInterpolationExtended)
				{
					for (int r = rStarti; r < rEndi; r++)
					{
						for (int c = cStarti; c < cEndi; c++)
						{
							ct = (c - tx);// (c - origin_x)
							rt = (r - ty);//(r - origin_y)

							x_real = ct*MI.at<double>(0, 0) + rt*MI.at<double>(0, 1) + centerSLAM.x;
							y_real = ct*MI.at<double>(1, 0) + rt*MI.at<double>(1, 1) + centerSLAM.y;

							//x_real = c*MI.at<double>(0, 0) + r*MI.at<double>(0, 1) + MI.at<double>(0, 2);
							//y_real = c*MI.at<double>(1, 0) + r*MI.at<double>(1, 1) + MI.at<double>(1, 2);


							x_real2 = c*MI.at<double>(0, 0) + r*MI.at<double>(0, 1) + MI.at<double>(0, 2);
							y_real2 = c*MI.at<double>(1, 0) + r*MI.at<double>(1, 1) + MI.at<double>(1, 2);

							float x2 = y_real;
							float y2 = x_real;

							float u = floor(x2);
							float v = floor(y2);

							float W1 = (u + 1 - x2)*(v + 1 - y2);
							float W2 = (x2 - u)*(v + 1 - y2);
							float W3 = (u + 1 - x2)*(y2 - v);
							float W4 = (x2 - u)*(y2 - v);

							if ((u >= 0 && u <  Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.rows - 1) && (v >= 0 && v <  (Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.cols - 1)))
							{
								float res = W1* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v) + W2* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v) + W3* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v + 1) + W4* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v + 1);
								dst_manual.at<float>(r, c) = res;
							}
							else
							{
								//dst_manual.at<float>(r, c) = 0.5;
							}
						}
					}


					rotateDegree = -22.5;

					//print rotate matrix
					M = getRotationMatrix2D(centerSLAM, rotateDegree, 1);
					MI;
					invertAffineTransform(M, MI);

					for (int r = 0; r < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM.rows; r++)
					{
						for (int c = 0; c < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM.cols; c++)
						{
							//float ct = (c - centerSLAM.x);// (c - origin_x)
							//float rt = (r - centerSLAM.y);//(r - origin_y)

							float ct = (c - tx);// (c - origin_x)
							float rt = (r - ty);//(r - origin_y)

							x_real = ct*MI.at<double>(0, 0) + rt*MI.at<double>(0, 1) + centerSLAM.x;
							y_real = ct*MI.at<double>(1, 0) + rt*MI.at<double>(1, 1) + centerSLAM.y;

							//x_real = c*MI.at<double>(0, 0) + r*MI.at<double>(0, 1) + MI.at<double>(0, 2);
							//y_real = c*MI.at<double>(1, 0) + r*MI.at<double>(1, 1) + MI.at<double>(1, 2);


							x_real2 = c*MI.at<double>(0, 0) + r*MI.at<double>(0, 1) + MI.at<double>(0, 2);
							y_real2 = c*MI.at<double>(1, 0) + r*MI.at<double>(1, 1) + MI.at<double>(1, 2);

							float x2 = y_real;
							float y2 = x_real;

							float u = floor(x2);
							float v = floor(y2);

							float W1 = (u + 1 - x2)*(v + 1 - y2);
							float W2 = (x2 - u)*(v + 1 - y2);
							float W3 = (u + 1 - x2)*(y2 - v);
							float W4 = (x2 - u)*(y2 - v);

							if ((u >= 0 && u <  Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.rows - 1) && (v >= 0 && v <  (Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.cols - 1)))
							{
								float res = W1* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v) + W2* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v) + W3* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v + 1) + W4* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v + 1);
								dst_manual.at<float>(r, c) = res;
							}
							else
							{
								//dst_manual.at<float>(r, c) = 0.5;
							}
						}
					}


					//rotateDegree = -75;

					////print rotate matrix
					//M = getRotationMatrix2D(centerSLAM, rotateDegree, 1);
					//MI;
					//invertAffineTransform(M, MI);

					//for (int r = 0; r < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM.rows; r++)
					//{
					//	for (int c = 0; c < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM.cols; c++)
					//	{
					//		float ct = (c - centerSLAM.x);// (c - origin_x)
					//		float rt = (r - centerSLAM.y);//(r - origin_y)

					//		x_real = ct*MI.at<double>(0, 0) + rt*MI.at<double>(0, 1) + centerSLAM.x;
					//		y_real = ct*MI.at<double>(1, 0) + rt*MI.at<double>(1, 1) + centerSLAM.y;

					//		float x2 = y_real;
					//		float y2 = x_real;

					//		float u = floor(x2);
					//		float v = floor(y2);

					//		float W1 = (u + 1 - x2)*(v + 1 - y2);
					//		float W2 = (x2 - u)*(v + 1 - y2);
					//		float W3 = (u + 1 - x2)*(y2 - v);
					//		float W4 = (x2 - u)*(y2 - v);

					//		if ((u >= 0 && u <  Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.rows - 1) && (v >= 0 && v <  Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.cols - 1))
					//		{
					//			float res = W1* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v) + W2* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v) + W3* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v + 1) + W4* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v + 1);

					//			if (dst_manual.at<float>(r + ty, c + tx) == 0 || dst_manual.at<float>(r + ty, c + tx) == 0.5)
					//				dst_manual.at<float>(r + ty, c + tx) = res;
					//			else
					//			{
					//				dst_manual.at<float>(r + ty, c + tx) = (dst_manual.at<float>(r + ty, c + tx) + res) / 2.0;
					//			}
					//		}
					//	}

					//}

					//rotateDegree = -90;

					////print rotate matrix
					//M = getRotationMatrix2D(centerSLAM, rotateDegree, 1);
					//MI;
					//invertAffineTransform(M, MI);

					//for (int r = 0; r < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM.rows; r++)
					//{
					//	for (int c = 0; c < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM.cols; c++)
					//	{
					//		float ct = (c - centerSLAM.x);// (c - origin_x)
					//		float rt = (r - centerSLAM.y);//(r - origin_y)

					//		x_real = ct*MI.at<double>(0, 0) + rt*MI.at<double>(0, 1) + centerSLAM.x;
					//		y_real = ct*MI.at<double>(1, 0) + rt*MI.at<double>(1, 1) + centerSLAM.y;

					//		float x2 = y_real;
					//		float y2 = x_real;

					//		float u = floor(x2);
					//		float v = floor(y2);

					//		float W1 = (u + 1 - x2)*(v + 1 - y2);
					//		float W2 = (x2 - u)*(v + 1 - y2);
					//		float W3 = (u + 1 - x2)*(y2 - v);
					//		float W4 = (x2 - u)*(y2 - v);

					//		if ((u >= 0 && u <  Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.rows - 1) && (v >= 0 && v <  Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.cols - 1))
					//		{
					//			float res = W1* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v) + W2* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v) + W3* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v + 1) + W4* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v + 1);

					//			if (dst_manual.at<float>(r + ty, c + tx) == 0 || dst_manual.at<float>(r + ty, c + tx) == 0.5)
					//				dst_manual.at<float>(r + ty, c + tx) = res;
					//			else
					//			{
					//				dst_manual.at<float>(r + ty, c + tx) = (dst_manual.at<float>(r + ty, c + tx) + res) / 2.0;
					//			}
					//		}
					//	}

					//}

					//cout << "hey";
				}
				else
				{
					//centerSLAM.x = round(Global_occupancyGridParams->middleColSLAM - Global_occupancyGridParams->middleCol);
					//centerSLAM.y = round(Global_occupancyGridParams->middleRowSLAM - Global_occupancyGridParams->totalRows);

					//centerSLAM.x = Global_occupancyGridParams->middleColSLAM;
					//centerSLAM.y = Global_occupancyGridParams->middleRowSLAM;

					rotateDegree = 0;
					centerSLAM.x = Global_occupancyGridParams->middleCol;
					centerSLAM.y = Global_occupancyGridParams->totalRows;
					tx = Global_occupancyGridParams->middleColSLAM;
					ty = Global_occupancyGridParams->middleRowSLAM;
					M = getRotationMatrix2D(centerSLAM, rotateDegree, 1);

					for (int r = 0; r < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.rows; r++)
					{
						for (int c = 0; c < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.cols; c++)
						{
							float ct = (c - centerSLAM.x);// (c - origin_x)
							float rt = (r - centerSLAM.y);//(r - origin_y)

							x_real2 = c*M.at<double>(0, 0) + r*M.at<double>(0, 1) + M.at<double>(0, 2);
							y_real2 = c*M.at<double>(1, 0) + r*M.at<double>(1, 1) + M.at<double>(1, 2);

							x_real = ct*t.at<double>(0, 0) + rt*t.at<double>(0, 1) + tx;
							y_real = ct*t.at<double>(1, 0) + rt*t.at<double>(1, 1) + ty;

							x_real3 = c*t.at<double>(0, 0) + r*t.at<double>(0, 1) + centerSLAM.x;
							y_real3 = c*t.at<double>(1, 0) + r*t.at<double>(1, 1) + centerSLAM.y;

							x = (int)round(x_real);//Nearest Neighbourhood interpolation
							y = (int)round(y_real);//Nearest Neighbourhood interpolation

							if (!bilinearInterpolation)
							{
								if (y >= 0 && y < dst_manual.rows)
								{
									if (x >= 0 && x < dst_manual.cols)
									{
										if (Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(r, c) > 0)
										{
											dst_manual.at<float>(y, x) = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(r, c);
											//cout << fixed << std::setprecision(2) << "(" << y << "," << x << ") -->" << Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(r, c) << "  " << endl;
										}
										else
										{
											//dst_manual.at<float>(y, x) = 0.5;
										}
									}
								}
							}
							else
							{
								x_real = ct*MI.at<double>(0, 0) + rt*MI.at<double>(0, 1) + centerSLAM.x;
								y_real = ct*MI.at<double>(1, 0) + rt*MI.at<double>(1, 1) + centerSLAM.y;

								float x2 = y_real;
								float y2 = x_real;

								float u = floor(x2);
								float v = floor(y2);

								float W1 = (u + 1 - x2)*(v + 1 - y2);
								float W2 = (x2 - u)*(v + 1 - y2);
								float W3 = (u + 1 - x2)*(y2 - v);
								float W4 = (x2 - u)*(y2 - v);

								if ((u >= 0 && u < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.rows - 1) && (v >= 0 && v < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.cols - 1))
								{
									float res = W1* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v) + W2* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v) + W3* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v + 1) + W4* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v + 1);
									dst_manual.at<float>(r + ty, c + tx) = res;
								}
								else
								{
									dst_manual.at<float>(r + ty, c + tx) = 0.5;
								}
							}
							//t = [cosd(teta) - sind(teta); sind(teta) cosd(teta)]; //matlab code
							//affine = [xscale*cosd(teta) xshear*sind(teta) xtrans; yshear*(-sind(teta)) yscale*cosd(teta) ytrans; 0 0 1]; //matlab code
							//xyz = affine*[(r - origin_x); (c - origin_y); 1]; //matlab code
							//xy = t*[(r - origin_x); (c - origin_y)];

						}
					}

					rotateDegree = 22.5;
					M = getRotationMatrix2D(centerSLAM, rotateDegree, 1);
					centerSLAM.x = Global_occupancyGridParams->middleCol;
					centerSLAM.y = Global_occupancyGridParams->middleRow;
					tx = Global_occupancyGridParams->middleColSLAM;
					ty = Global_occupancyGridParams->middleRowSLAM;
					M = getRotationMatrix2D(centerSLAM, rotateDegree, 1);

					t.at<double>(0, 0) = cos(rotateDegree * CV_PI / 180.0);//cos(teta)
					t.at<double>(0, 1) = sin(rotateDegree * CV_PI / 180.0);//sin(teta)
					t.at<double>(0, 2) = centerSLAM.y;//y trans
					t.at<double>(1, 0) = -sin(rotateDegree * CV_PI / 180.0);//-sin(teta)
					t.at<double>(1, 1) = cos(rotateDegree * CV_PI / 180.0);//cosd(teta)
					t.at<double>(1, 2) = centerSLAM.x;//x trans

					for (int r = 0; r < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.rows; r++)
					{
						for (int c = 0; c < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.cols; c++)
						{
							float ct = (c - centerSLAM.x);// (c - origin_x)
							float rt = (r - centerSLAM.y);//(r - origin_y)

							x_real2 = c*M.at<double>(0, 0) + r*M.at<double>(0, 1) + M.at<double>(0, 2);
							y_real2 = c*M.at<double>(1, 0) + r*M.at<double>(1, 1) + M.at<double>(1, 2);

							x_real = ct*t.at<double>(0, 0) + rt*t.at<double>(0, 1) + tx;
							y_real = ct*t.at<double>(1, 0) + rt*t.at<double>(1, 1) + ty;

							x_real3 = c*t.at<double>(0, 0) + r*t.at<double>(0, 1) + centerSLAM.x;
							y_real3 = c*t.at<double>(1, 0) + r*t.at<double>(1, 1) + centerSLAM.y;

							x = (int)round(x_real);//Nearest Neighbourhood interpolation
							y = (int)round(y_real);//Nearest Neighbourhood interpolation

							if (!bilinearInterpolation)
							{
								if (y >= 0 && y < dst_manual.rows)
								{
									if (x >= 0 && x < dst_manual.cols)
									{
										if (Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(r, c) > 0)
										{
											dst_manual.at<float>(y, x) = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(r, c);
											//cout << fixed << std::setprecision(2) << "(" << y << "," << x << ") -->" << Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(r, c) << "  " << endl;
										}
										else
										{
											dst_manual.at<float>(y, x) = 0.5;
										}
									}
								}
							}
							else
							{
								x_real = ct*MI.at<double>(0, 0) + rt*MI.at<double>(0, 1) + centerSLAM.x;
								y_real = ct*MI.at<double>(1, 0) + rt*MI.at<double>(1, 1) + centerSLAM.y;

								float x2 = y_real;
								float y2 = x_real;

								float u = floor(x2);
								float v = floor(y2);

								float W1 = (u + 1 - x2)*(v + 1 - y2);
								float W2 = (x2 - u)*(v + 1 - y2);
								float W3 = (u + 1 - x2)*(y2 - v);
								float W4 = (x2 - u)*(y2 - v);

								if ((u >= 0 && u < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.rows - 1) && (v >= 0 && v < Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.cols - 1))
								{
									float res = W1* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v) + W2* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v) + W3* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u, v + 1) + W4* Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(u + 1, v + 1);
									dst_manual.at<float>(r + ty, c + tx) = res;
								}
								else
								{
									dst_manual.at<float>(r + ty, c + tx) = 0.5;
								}
							}
							//t = [cosd(teta) - sind(teta); sind(teta) cosd(teta)]; //matlab code
							//affine = [xscale*cosd(teta) xshear*sind(teta) xtrans; yshear*(-sind(teta)) yscale*cosd(teta) ytrans; 0 0 1]; //matlab code
							//xyz = affine*[(r - origin_x); (c - origin_y); 1]; //matlab code
							//xy = t*[(r - origin_x); (c - origin_y)];

						}
					}
				}

				//cout << "hey";
#endif
#endif

#ifdef WITH_SLAM
				if (Global_occupancyGridParams->useBilinearInterpolation)
					Global_occupancyGridGlobals->occupancyGridMap.set_OccupancyGridSLAMMatrix_ByBilinearInterpolation(*Global_occupancyGridParams);

				//float zr = 60;
				//float xr = 117;


				////		float z = Global_occupancyGridParams->totalRows - (obj.Zw / Global_occupancyGridParams->ZWorldResolution) - 1;
				////		float x = (obj.Xw / Global_occupancyGridParams->XWorldResolution) + Global_occupancyGridParams->middleCol;	

				//float zw = -(zr - Global_occupancyGridParams->totalRows + 1) * Global_occupancyGridParams->ZWorldResolution;
				//float xw = (xr - Global_occupancyGridParams->middleCol) * Global_occupancyGridParams->XWorldResolution;
				//float dr = sqrt(pow(xw, 2) + pow(zw, 2));

				//std::cout << "z:" << zr << " x:" << xr << " zw:" << zw << " xw:" << xw << " dr:" << dr << endl;

				//drawObjects on SLAM color
				Mat ColorSLAM;
				cv::cvtColor(Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM, ColorSLAM, CV_GRAY2RGB);

				Mat ColorSLAM_Thresholded;
				cv::cvtColor(Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM_Thresholded, ColorSLAM_Thresholded, CV_GRAY2RGB);

				Mat ColorOcc;
				cv::cvtColor(Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix, ColorOcc, CV_GRAY2RGB);

				int Global_currentIMGNum = Global_minIMGIndex + Global_currentIMGIndex * Global_imgIncrement;
				int mult = 2;

				if (Global_currentIMGNum == 19)
					addDrawPoint(19, "s1", 105, 104, 13, 5, ColorOcc, ColorSLAM, 4.3);


				if (Global_currentIMGNum == 29)
				{
					addDrawPoint(29, "re", 100, 114, 12, 4, ColorOcc, ColorSLAM, 4.97);
					addDrawPoint(29, "c1", 115, 98, 5, 3, ColorOcc, ColorSLAM, 3.34);
				}

				//if (Global_currentIMGNum == 39)
				//{
				//	addDrawPoint(39, "d1", 59, 114, 7, 4, ColorOcc, ColorSLAM, 8.8);
				//}

				if (Global_currentIMGNum == 49)
				{
					addDrawPoint(49, "y1", 92, 123, 6, 3, ColorOcc, ColorSLAM, 6.14);
					addDrawPoint(49, "d1", 58, 81, 41, 5, ColorOcc, ColorSLAM, 8.8);
				}

				//if (Global_currentIMGNum == 79)
				//{
				//	addDrawPoint(79, "sl2", 112, 105, 6, 3, ColorOcc, ColorSLAM, 3.77);
				//	addDrawPoint(79, "sr2", 115, 113, 6, 3, ColorOcc, ColorSLAM, 3.77);
				//}


				if (Global_currentIMGNum == 89)
				{
					addDrawPoint(89, "s2", 113, 87, 13, 3, ColorOcc, ColorSLAM, 3.77);
				}

				//if (Global_currentIMGNum == 109)
				//{
				//	addDrawPoint(86, "s2", 92, 122, 7, 3, ColorOcc, ColorSLAM, 3.77);
				//}

				if (Global_currentIMGNum == 119)
				{
					addDrawPoint(86, "c2", 88, 100, 6, 4, ColorOcc, ColorSLAM, 6.05);
				}

				if (Global_currentIMGNum == 129)
				{
					addDrawPoint(86, "d2", 51, 72, 27, 6, ColorOcc, ColorSLAM, 9.2);
				}

				if (Global_currentIMGNum == 139)
				{
					addDrawPoint(86, "sa", 99, 110, 8, 3, ColorOcc, ColorSLAM, 4.95);
					addDrawPoint(86, "s3", 63, 111, 10, 7, ColorOcc, ColorSLAM, 8.35);
				}

				for (int i = 0; i < drawings.size(); i++)
				{
					cv::rectangle(ColorSLAM, drawings[i]->p1, drawings[i]->p2, cv::Scalar(0, 255, 0), 1 * mult);
					cv::rectangle(ColorSLAM_Thresholded, drawings[i]->p1, drawings[i]->p2, cv::Scalar(0, 255, 0), 1 * mult);

					ostringstream dt;
					dt << fixed << std::setprecision(2) << " - " << drawings[i]->distance << " (%" << drawings[i]->Pou << ")";


					if (i % 4 == 0)
					{
						Point2i textPoint(drawings[i]->p1.x + 5 * mult, drawings[i]->p1.y);
						cv::putText(ColorSLAM, drawings[i]->text, drawings[i]->p1, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(0, 0, 255), 0.35*mult, CV_AA);
						cv::putText(ColorSLAM, dt.str().c_str(), textPoint, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(0, 0, 255), 0.35*mult, CV_AA);

						cv::putText(ColorSLAM_Thresholded, drawings[i]->text, drawings[i]->p1, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(0, 0, 255), 0.35*mult, CV_AA);
						cv::putText(ColorSLAM_Thresholded, dt.str().c_str(), textPoint, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(0, 0, 255), 0.35*mult, CV_AA);
					}
					else
					{
						if (i % 4 == 1)
						{
							cv::putText(ColorSLAM, drawings[i]->text, drawings[i]->p2, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(255, 0, 0), 0.35*mult, CV_AA);
							cv::putText(ColorSLAM_Thresholded, drawings[i]->text, drawings[i]->p2, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(255, 0, 0), 0.35*mult, CV_AA);
							Point2i textPoint(drawings[i]->p2.x + 5 * mult, drawings[i]->p2.y);
							cv::putText(ColorSLAM, dt.str().c_str(), textPoint, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(255, 0, 0), 0.35*mult, CV_AA);
							cv::putText(ColorSLAM_Thresholded, dt.str().c_str(), textPoint, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(255, 0, 0), 0.35*mult, CV_AA);
						}
						else
						{
							if (i % 4 == 2)
							{
								cv::putText(ColorSLAM, drawings[i]->text, drawings[i]->p1, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(0, 255, 0), 0.35*mult, CV_AA);
								cv::putText(ColorSLAM_Thresholded, drawings[i]->text, drawings[i]->p1, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(0, 255, 0), 0.35*mult, CV_AA);
								Point2i textPoint(drawings[i]->p1.x + 5 * mult, drawings[i]->p1.y);
								cv::putText(ColorSLAM, dt.str().c_str(), textPoint, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(0, 255, 0), 0.35*mult, CV_AA);
								cv::putText(ColorSLAM_Thresholded, dt.str().c_str(), textPoint, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(0, 255, 0), 0.35*mult, CV_AA);
							}
							else
							{
								cv::putText(ColorSLAM, drawings[i]->text, drawings[i]->p2, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(255, 0, 255), 0.35*mult, CV_AA);
								cv::putText(ColorSLAM_Thresholded, drawings[i]->text, drawings[i]->p2, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(255, 0, 255), 0.35*mult, CV_AA);
								Point2i textPoint(drawings[i]->p2.x + 5 * mult, drawings[i]->p2.y);
								cv::putText(ColorSLAM, dt.str().c_str(), textPoint, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(255, 0, 255), 0.35*mult, CV_AA);
								cv::putText(ColorSLAM_Thresholded, dt.str().c_str(), textPoint, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.50*mult, cv::Scalar(255, 0, 255), 0.35*mult, CV_AA);
							}

						}

					}

					Point2i closest_P1(drawings[i]->closes_Point.x - 1, drawings[i]->closes_Point.y - 1);
					Point2i closest_P2(drawings[i]->closes_Point.x + 1, drawings[i]->closes_Point.y + 1);

					cv::rectangle(ColorSLAM, closest_P1, closest_P2, cv::Scalar(0, 255, 255), 1 * mult);
					cv::rectangle(ColorSLAM_Thresholded, closest_P1, closest_P2, cv::Scalar(0, 255, 255), 1 * mult);

				}

				//cout << "";

#endif
				//for (size_t obj_index = 0; obj_index < Global_mergedObjectList2.size(); obj_index++)
				//{
				//	MyObject obj = Global_mergedObjectList2[obj_index];
				//	if ((!obj.isDeleted) && (obj.Zw < Global_occupancyGridParams->ZWorldLimit) && (abs(obj.Xw) < Global_occupancyGridParams->XWorldLimit) && (obj.Yw > 0.5))
				//	{
				//		int objLeftd, objLeftv, objRightd, objRightv, objLeftu, objRightu;
				//		float objLeft_Xw, objLeft_Yw, objLeft_Zw, objRight_Xw, objRight_Yw, objRight_Zw;

				//		if (obj.isLeftLeaning)
				//		{
				//			objLeftu = obj.xLow_Index;
				//			objRightu = obj.xHigh_Index;
				//			objLeftd = obj.yhighDisp_Index;
				//			objLeftv = obj.yhighDisp_Vmin_Index;
				//			objRightd = obj.ylowDisp_Index;
				//			objRightv = obj.ylowDisp_Vmin_Index;
				//		}
				//		else
				//		{
				//			objLeftu = obj.xLow_Index;
				//			objRightu = obj.xHigh_Index;
				//			objLeftd = obj.ylowDisp_Index;
				//			objLeftv = obj.ylowDisp_Vmin_Index;
				//			objRightd = obj.yhighDisp_Index;
				//			objRightv = obj.yhighDisp_Vmin_Index;
				//		}
				//								
				//		ImageToWorldCoordinates(objLeftu, objLeftv, objLeftd, objLeft_Xw, objLeft_Yw, objLeft_Zw);
				//		ImageToWorldCoordinates(objRightu, objRightv, objRightd, objRight_Xw, objRight_Yw, objRight_Zw);

				//		float z_Left = Global_occupancyGridParams->totalRows - (objLeft_Zw / Global_occupancyGridParams->ZWorldResolution) - 1;
				//		float x_Left = (objLeft_Xw / Global_occupancyGridParams->XWorldResolution) + Global_occupancyGridParams->middleCol;
				//		cv::Point2i oldPoint_Left((int)x_Left, (int)z_Left);						
				//		cv::Point2i newPoint_left = Global_occupancyGridParams->setMapCoordinatesByNN(oldPoint_Left.x, oldPoint_Left.y);

				//		float z_Right = Global_occupancyGridParams->totalRows - (objRight_Zw / Global_occupancyGridParams->ZWorldResolution) - 1;
				//		float x_Right = (objRight_Xw / Global_occupancyGridParams->XWorldResolution) + Global_occupancyGridParams->middleCol;
				//		cv::Point2i oldPoint_Right((int)x_Right, (int)z_Right);
				//		cv::Point2i newPoint_Right = Global_occupancyGridParams->setMapCoordinatesByNN(oldPoint_Right.x, oldPoint_Right.y);

				//		cv::line(ColorOcc, oldPoint_Left, oldPoint_Right, cv::Scalar(255, 0, 255), 1);
				//		cv::line(ColorSLAM, newPoint_left, newPoint_Right, cv::Scalar(255, 0, 255), 1);

				//		float z = Global_occupancyGridParams->totalRows - (obj.Zw / Global_occupancyGridParams->ZWorldResolution) - 1;
				//		float x = (obj.Xw / Global_occupancyGridParams->XWorldResolution) + Global_occupancyGridParams->middleCol;						

				//		cv::Point2i oldPoint((int)x, (int)z);
				//		cv::circle(ColorOcc, oldPoint, 3, cv::Scalar(255, 0, 0));
				//		cv::Point2i new_xy = Global_occupancyGridParams->setMapCoordinatesByNN(x, z);
				//		cv::circle(ColorSLAM, new_xy, 3, cv::Scalar(255, 0, 0));

				//		std::cout << "index:" << obj_index << " hi_d:" << obj.yhighDisp_Index << " low_d:" << obj.ylowDisp_Index << " hir_d:" << obj.yhighDisp_Index_real << " lowr_d:" << obj.ylowDisp_Index_real   << " dd:" << obj.directDistance << " rd:" << obj.realDistanceTiltCorrected << " Zw:" << obj.Zw << " Xw:" << obj.Xw << " Yw:" << obj.Yw << " W:" << obj.objWidth << " oX:" << (int)x << " oZ:" << (int)z << " sX:" << new_xy.x << " sZ:" << new_xy.y<< std::endl;
				//		std::cout << "index:" << obj_index << " ll:" << obj.isLeftLeaning << " lu:" << objLeftu << " lv:" << objLeftv  << " ld:" << objLeftd << " ru:" << objRightu << " rv:" << objRightv << " rd:" << objRightd << " xl:" << (int)x_Left << " zl:" << (int)z_Left << " xr:" << (int)x_Right << " zr:" << (int)z_Right << endl << endl;
				//	}
				//}

#ifdef WITH_DOCK_DETECTION
				Mat line_draws;
				getBoundingBoxes(Global_Current_Left_Image, dispD, drawed, Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix, line_draws);

#endif

				Global_uDisparityGlobals->clear();
				Global_vDisparityGlobals->clear();
				Global_dDisparityGlobals->clear();

#ifdef WITH_OCCUPANCY_GRID
				Mat check_occupancy = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix;
#endif

#ifdef WITH_SLAM
				Mat check_occupancy_SLAM = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM;
				Mat check_occupancy_SLAM_Thresholded = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM_Thresholded;
#else
#ifdef WITH_OCCUPANCY_GRID
				Mat check_occupancy_thresholded = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix_Thresholded;
#endif
#endif

				if (showImages)
				{
					//cv::imshow("Window1", Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix);
					//cv::imshow("Window1", Global_uDisparityGlobals->uDisparityImage);
					cv::imshow("Window1", uDispDraw);

					cv::imshow("Window4", Global_Current_DenseDisparity_Image);
					//cv::imshow("Window6", colorV);


					//cv::imshow("Window1", Global_uDisparityGlobals->uDisparityImage);
					//cv::imshow("Window2", Global_uDisparityGlobals->uDisparityImage);
					//cv::imshow("Window3", line_draws);
					//cv::imshow("Window4", colorV);
					//cv::imshow("Window5", Global_uDisparityGlobals->uDisparityImage_thresholded);
#ifdef WITH_OCCUPANCY_GRID
					//cv::imshow("Window6", Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix);
					cv::imshow("Window3", ColorOcc);

#ifdef WITH_SLAM
					//cv::imshow("Window7", Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM);
					cv::imshow("Window2", ColorSLAM_Thresholded);

#else
					cv::imshow("Window7", drawed);
#endif

#endif
#ifdef WITH_LANE_DETECTION
					Mat cannyResized;
					int width = (detLanes->dstCanny.cols + 1) * 0.75;
					int height = (detLanes->dstCanny.rows + 1) * 0.75;
					cv::resize(detLanes->dstCanny, cannyResized, cv::Size(width, height));
					cv::imshow("Window6", Global_Current_DenseDisparity_Image);
					cv::imshow("Window7", cannyResized);
#endif
					//cv::imshow("Window8", Global_Current_DenseDisparity_Image);

					int c = (cvWaitKey(1000) & 0xff);
					if (c == 32)
					{
						c = (cvWaitKey(1000) & 0xff);
						while (c != 32)
						{
							c = (cvWaitKey(3000) & 0xff);
						}
					}//cvWaitKey if
				}//if show images
			}//Offline Imagelist For loop

			 // Close the camera
			camera.StopCapture();
			camera.Disconnect();

			// Destroy the Triclops context
			triclops_status = triclopsDestroyContext(context);

			std::cout << "Closed" << std::endl;
		}//if not using GPU
		else//if use GPU
		{
#ifdef WITH_GPU
			while (strcmp(err.c_str(), "Ok.") == 0)
			{
				fc2Error = camera.RetrieveBuffer(&grabbedImage);
				errC = fc2Error.GetDescription();
				err = errC;

				ImageContainer imageCont;
				generatemonoStereoPair(context, grabbedImage, imageCont, monoStereoPair);

				// Prepare context for frame rectification
				triclops_status = triclopsPrepareRectificationData(context, bumblebeeObject->outputHeight, bumblebeeObject->outputWidth, monoStereoPair.right.nrows, monoStereoPair.right.ncols);//could be beneficial in multi CPU code or paralel computing
																																																 //Rectify
				triclops_status = triclopsRectify(context, &monoStereoPair);
				//triclops_examples::handleError("triclopsRectify()", triclops_status, __LINE__);

				TriclopsImage imageL, imageR;
				triclops_status = triclopsGetImage(context, TriImg_RECTIFIED, TriCam_RIGHT, &imageR);
				triclops_status = triclopsGetImage(context, TriImg_RECTIFIED, TriCam_LEFT, &imageL);
				//triclops_examples::handleError("triclopsGetImage()", triclops_status, __LINE__);

				Global_currentIMGIndex = 0;
				Global_Current_Left_Image = convertTriclops2OpencvMat(imageL);
				Global_Current_Right_Image = convertTriclops2OpencvMat(imageR);

				setDenseDisparityMapCUDA();
				set_UVDisparityMaps();
				set_BestLine_From_VDisparity();
				set_ImageInfinityPoint_And_VehicleRoadLine(Global_imgMiddleCol, Global_vDisparityGlobals->bestRoadLine_vIntersection, Global_imgTotalCols, Global_imgTotalRows);

				set_LabelsOf_UDisparityLines_Using_LabelingAlg();
				//cv::Mat draw1 = drawUDispObjects(uDispDraw1, Global_labeledObjectList);
				merge_MaskVObjects_From_UDisparity_By_Labeling_Alg(Global_labeledObjectList);
				//cv::Mat draw2 = drawUDispObjects(uDispDraw2, Global_mergedObjectList1);
				merge_InclinedObjects_From_UDisparity_By_Labeling_Alg(Global_mergedObjectList1);
				//cv::Mat draw3 = drawUDispObjects(uDispDraw3, Global_mergedObjectList2);
				set_SideBuildingsIfAny_And_Filter(Global_mergedObjectList2);
				createDenseDispObjects(Global_mergedObjectList2);

#ifdef WITH_LANE_DETECTION
				laneDetection *detLanes = new laneDetection(1.05, Global_Current_Left_Image);
				detLanes->testLanes(true, Global_Current_Left_Image);
				cv::imshow("Window1", Global_Current_Left_Image);
				int c = (cvWaitKey(1000) & 0xff);
				Mat laneIMG = detLanes->getLanes(false, Global_vDisparityGlobals->bestRoadLine_vIntersection);
#endif	

				//cv::Mat draw4 = drawUDispObjects(uDispDraw4, Global_mergedObjectList2);		
				cv::Mat colorV;
				cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV, CV_GRAY2BGR);
				cv::line(colorV, Point(0, Global_vDisparityGlobals->bestRoadLine_vIntersection), Point(Global_vDisparityGlobals->bestRoadLine_dInterSection, Global_vDisparityParams->totalRows - 1), cv::Scalar(0, 0, 255));
				cv::line(colorV, Point(Global_vDisparityGlobals->piecewiseBestLine[0], Global_vDisparityGlobals->piecewiseBestLine[1]), Point(Global_vDisparityGlobals->piecewiseBestLine[2], Global_vDisparityGlobals->piecewiseBestLine[3]), cv::Scalar(0, 255, 0));

				cv::Mat drawed = drawObjects(Global_Current_Left_Image, Global_mergedObjectList2);
				cv::Mat uDispDraw = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, drawed.type());
				uDispDraw = drawUDispObjects(uDispDraw, Global_mergedObjectList2);

				int deleted = getNumObjectsDeleted(Global_mergedObjectList2);
				int livingObj = Global_currentNumOfObjects - deleted;
				//diff_disparity = (static_cast<float>(t2)-static_cast<float>(t1));
				//cout << "NumOfObjs:" << Global_currentNumOfObjects << " LivingObjects:" << livingObj << " time: " << diff_disparity << "ms." << endl;

#ifdef WITH_OCCUPANCY_GRID				
				set_occupancyUDisparityMap();
#endif

				Global_uDisparityGlobals->clear();
				Global_vDisparityGlobals->clear();
				Global_dDisparityGlobals->clear();

				if (showImages)
				{
					cv::imshow("Window1", Global_uDisparityGlobals->uDisparityImage);
					cv::imshow("Window2", uDispDraw);
					cv::imshow("Window3", drawed);
					cv::imshow("Window4", colorV);
					cv::imshow("Window5", Global_uDisparityGlobals->uDisparityImage_thresholded);
#ifdef WITH_OCCUPANCY_GRID
					cv::imshow("Window6", Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix);
#endif
#ifdef WITH_LANE_DETECTION
					Mat cannyResized;
					int width = (detLanes->dstCanny.cols + 1) * 0.75;
					int height = (detLanes->dstCanny.rows + 1) * 0.75;
					cv::resize(detLanes->dstCanny, cannyResized, cv::Size(width, height));
					cv::imshow("Window6", Global_Current_DenseDisparity_Image);
					cv::imshow("Window7", cannyResized);
#endif
					cv::imshow("Window8", Global_Current_DenseDisparity_Image);

					int c = (cvWaitKey(1000) & 0xff);
					if (c == 32)
					{
						c = (cvWaitKey(1000) & 0xff);
						while (c != 32)
						{
							c = (cvWaitKey(3000) & 0xff);
						}
					}//cvWaitKey if
				}//if show images
			}

			// Close the camera
			camera.StopCapture();
			camera.Disconnect();

			// Destroy the Triclops context
			triclops_status = triclopsDestroyContext(context);

			std::cout << "Closed" << std::endl;
#else
			cout << "You should Define GPU...(#define WITH_GPU)" << endl;
#endif
		}//else use GPU
	}
#pragma endregion Online
}

#ifdef WITH_GPU
inline void StereoImageBuffer::setDenseDisparityMapCUDA()
{
	int iters = 5;
	int levels = 5;
	int plane = 4;
	int msg_type = 5;

	cv::cuda::GpuMat imgLeft_GPU;
	cv::cuda::GpuMat imgRight_GPU;

	imgLeft_GPU.upload(Global_Current_Left_Image);
	imgRight_GPU.upload(Global_Current_Right_Image);

	auto sbm_constant = cv::cuda::createStereoConstantSpaceBP(Global_numDisparities, iters, levels, plane, msg_type);

	auto imgDisparity16S = cv::Mat(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_16SC1);
	auto imgDisparity8U = cv::Mat(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_8UC1);
	auto imgDisparity16S_GPU = cv::cuda::GpuMat(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_16SC1);

	sbm_constant->compute(imgLeft_GPU, imgRight_GPU, imgDisparity16S_GPU);
	imgDisparity16S_GPU.download(imgDisparity16S);

	imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 1.0);

	Global_Current_DenseDisparity_Image = imgDisparity8U;
}
#endif

inline void StereoImageBuffer::set_DenseDisparityMap()
{
	int uniquenessRatio = 1;
	int disp12MaxDiff = -1;
	int preFilterCap = 11;
	int mode = 0;
	int blockSize = 17;

	int P1 = 8 * 1 * blockSize*blockSize;//BlockSize = 19: 2888 - BlockSize = 15: 1800
	int P2 = 32 * 1 * blockSize*blockSize;//BlockSize = 19: 11552 - BlockSize = 15: 7200

	int min_disparity = 0;
	auto ssgbm = cv::StereoSGBM::create(min_disparity, Global_numDisparities  /*16X*/, blockSize /*3..11*/);

	//auto imgDisparity16S = cv::Mat(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_16S);	
	auto imgDisparity8U = cv::Mat(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_8UC1);
	//auto imgDisparity8U2 = cv::Mat(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_8UC1);

	auto imgDisparity32F = cv::Mat(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_32FC1); //check float numbers
																													  //auto imgDisparity32F2 = cv::Mat(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_32FC1); //check float numbers

	if (mode == 1) ssgbm->setMode(StereoSGBM::MODE_HH);
	else ssgbm->setMode(StereoSGBM::MODE_SGBM);

	ssgbm->setUniquenessRatio(uniquenessRatio);
	ssgbm->setPreFilterCap(preFilterCap);
	ssgbm->setDisp12MaxDiff(disp12MaxDiff);
	ssgbm->setP1(P1);
	ssgbm->setP2(P2);
	//ssgbm->compute(Global_Current_Left_Image, Global_Current_Right_Image, imgDisparity16S);
	ssgbm->compute(Global_Current_Left_Image, Global_Current_Right_Image, imgDisparity32F);

	//ssgbm->setSpeckleRange(1);
	//ssgbm->setSpeckleWindowSize(1);

	double ratio;
	////-- Check its Extreme values //removed since opencv  containing disparity values scaled by 16.
	//double minVal; double maxVal;
	//cv::minMaxLoc(imgDisparity16S, &minVal, &maxVal);
	//ratio = (Global_numDisparities - 1) / (maxVal - minVal);

	ratio = 1. / 16.;

	int currentIMGNum = Global_minIMGIndex + Global_currentIMGIndex * Global_imgIncrement;
	ostringstream text;
	text << currentIMGNum;

	string Global_currentIMGNum_str;
	Global_currentIMGNum_str = text.str();

	if (currentIMGNum < 100) Global_currentIMGNum_str = "0000000" + Global_currentIMGNum_str;
	if (currentIMGNum > 100) Global_currentIMGNum_str = "000000" + Global_currentIMGNum_str;

	//Mat leftIMG = Global_Current_Left_Image;
	//Mat rightIMG = Global_Current_Right_Image;
	//Mat rightIMG_Rec = cv::imread("E:\\KUDiary\\Important Doc\\3D Detection\\stereo Images\\Bumblebee\\sancaktepe2R_wide_res1280X960_64bit_17b\\imgleft" + Global_currentIMGNum_str + ".pgm", CV_LOAD_IMAGE_UNCHANGED);
	//Mat leftIMG_Rec = cv::imread("E:\\KUDiary\\Important Doc\\3D Detection\\stereo Images\\Bumblebee\\sancaktepe2R_wide_res1280X960_64bit_17b\\imgright" + Global_currentIMGNum_str + ".pgm", CV_LOAD_IMAGE_UNCHANGED);
	//Mat disparityOpenCVIMG_Org = cv::imread("E:\\KUDiary\\Important Doc\\3D Detection\\stereo Images\\Bumblebee\\sancaktepe2R_wide_res1280X960_64bit_17b\\disp_opencv" + Global_currentIMGNum_str + ".pgm", CV_LOAD_IMAGE_UNCHANGED);
	//Mat disparity16S = cv::imread("E:\\KUDiary\\Important Doc\\3D Detection\\stereo Images\\Bumblebee\\sancaktepe2R_wide_res1280X960_64bit_17b\\disp" + Global_currentIMGNum_str + ".pgm", CV_LOAD_IMAGE_UNCHANGED);
	////text.str("");
	////text.clear();

	////-- 4. Display it as a CV_8UC1 image
	//imgDisparity32F.convertTo(imgDisparity32F2, CV_32FC1, ratio);
	//imgDisparity32F.convertTo(imgDisparity8U2, CV_8UC1, ratio);
	imgDisparity32F.convertTo(Global_Current_DenseDisparity_Image32F, CV_32FC1, ratio);
	Global_Current_DenseDisparity_Image32F.convertTo(Global_Current_DenseDisparity_Image, CV_8UC1, 1.0);

	//double minVal; double maxVal;
	//cv::minMaxLoc(Global_Current_DenseDisparity_Image32F, &minVal, &maxVal);
	//std::cout << "min: " << minVal << " max:" << maxVal << " (mindisp:" << min_disparity << " maxdisp:" << Global_numDisparities << ")" << std::endl;

	//imgDisparity8U = Global_Current_DenseDisparity_Image;
	//Mat imgDisparity32F_rounded = Global_Current_DenseDisparity_Image32F;
	//cout << "" ;

	//Global_Current_DenseDisparity_Image = imgDisparity8U;
}

inline void StereoImageBuffer::set_UDisparityMap()
{
	cv::Mat uDisparityMap = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, Global_Current_DenseDisparity_Image.type());

	int d;
	const unsigned char* cell_disparity;
	unsigned char* cell_uDisparity;

	for (int j = 0; j < Global_dDisparityParams->totalCols; j++)
	{
		for (int i = 0; i< Global_dDisparityParams->totalRows; i++)
		{
			cell_disparity = Global_Current_DenseDisparity_Image.ptr<const unsigned char>(i, j);
			d = cell_disparity[0];

			cell_uDisparity = uDisparityMap.ptr<unsigned char>(d); //continuous ROW
			cell_uDisparity[j] += 1;
		}
	}

	Global_uDisparityGlobals->uDisparityImage = uDisparityMap;
}

inline void StereoImageBuffer::set_VDisparityMap()
{
	cv::Mat vDisparityMap = cv::Mat::zeros(Global_vDisparityParams->totalRows, Global_vDisparityParams->totalCols, Global_Current_DenseDisparity_Image.type());

	int d;
	const unsigned char* row_disparity;
	unsigned char* cell_vDisparity;

	for (int v = 0; v< Global_dDisparityParams->totalRows; v++)
	{
		row_disparity = Global_Current_DenseDisparity_Image.ptr<const unsigned char>(v);

		for (int j = 0; j < Global_dDisparityParams->totalCols; j++)
		{
			d = row_disparity[j];

			cell_vDisparity = vDisparityMap.ptr<unsigned char>(v, d);
			cell_vDisparity[0] += 1;
		}
	}

	Global_vDisparityGlobals->vDisparityImage = vDisparityMap;
}

inline void StereoImageBuffer::set_UVDisparityMaps()
{
	cv::Mat uDisparityMap = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, Global_Current_DenseDisparity_Image.type());
	cv::Mat vDisparityMap = cv::Mat::zeros(Global_vDisparityParams->totalRows, Global_vDisparityParams->totalCols, Global_Current_DenseDisparity_Image.type());

	const unsigned char* d;
	int v, u;

	for (v = 0; v< Global_dDisparityParams->totalRows; v++)
	{
		d = Global_Current_DenseDisparity_Image.ptr<const unsigned char>(v);

		for (u = 0; u < Global_dDisparityParams->totalCols; u++)
		{
			vDisparityMap.at<uchar>(v, (d[u]))++;
			uDisparityMap.at<uchar>((d[u]), u)++;
		}
	}

	Global_vDisparityGlobals->vDisparityImage = vDisparityMap;
	Global_uDisparityGlobals->uDisparityImage = uDisparityMap;
}

inline void StereoImageBuffer::set_BestLine_From_VDisparity() const
{
	cv::Range shifted_rowRange;
	cv::Range shifted_colRange;

	//find the line boundaries and the line equation parameters
	int v = Global_vDisparityParams->totalRows - 1;
	int d = Global_vDisparityParams->totalCols - 1;

	float maxLinelength = 0;
	float currentLineLength, currentLineAngleRadian, currentLineAngleDegree, currentLine_m;
	float x1, y1, x2, y2;
	bool isFound = false;
	int maxLineIndex = 0;

	//search in the shifted row and col to get faster
	shifted_rowRange.start = Global_vDisparityParams->shiftedStartRangeV;
	shifted_rowRange.end = Global_vDisparityParams->totalRows;
	shifted_colRange.start = Global_vDisparityParams->shiftedStartRangeD;
	shifted_colRange.end = Global_vDisparityParams->totalCols;

	Global_vDisparityGlobals->vDisparityImage_shifted = Global_vDisparityGlobals->vDisparityImage(shifted_rowRange, shifted_colRange);
	//search in the shifted row and col to get faster		

	cv::GaussianBlur(Global_vDisparityGlobals->vDisparityImage_shifted, Global_vDisparityGlobals->vDisparityImage_blurred, Global_vDisparityParams->gaussianBlur_size, Global_vDisparityParams->gaussianBlur_sigmaX, Global_vDisparityParams->gaussianBlur_sigmaY, Global_vDisparityParams->gaussianBlur_borderType);
	cv::threshold(Global_vDisparityGlobals->vDisparityImage_blurred, Global_vDisparityGlobals->vDisparityImage_thresholded, Global_vDisparityParams->thresholding_threshold, Global_vDisparityParams->thresholding_maxVal, Global_vDisparityParams->thresholding_type);
	cv::HoughLinesP(Global_vDisparityGlobals->vDisparityImage_thresholded, Global_vDisparityGlobals->allLines, Global_vDisparityParams->houghLinesP_rho, Global_vDisparityParams->houghLinesP_theta, Global_vDisparityParams->houghLinesP_threshold, Global_vDisparityParams->houghLinesP_minLineLength, Global_vDisparityParams->houghLinesP_maxLineGap);

	int totalNumOfLines = Global_vDisparityGlobals->allLines.size();

	//Mat vDisp = Global_vDisparityGlobals->vDisparityImage;
	//Mat vDisp_shift = Global_vDisparityGlobals->vDisparityImage_shifted;
	//Mat vDisp_thresholded = Global_vDisparityGlobals->vDisparityImage_thresholded;

	//cv::Mat colorV_All_Lines;
	//cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV_All_Lines, CV_GRAY2BGR);

	//Create real baseline
	//float Zs, Ys_v0, vBase, Yw = 0;
	//float fx = stereoParams.f / stereoParams.sx; //focal_length_pixels in X coordinate
	//float fy = stereoParams.f / stereoParams.sy; //focal_length_pixels in Y coordinate
	//Global_vDisparityGlobals->vDisparityImage_Baseline = cv::Mat::zeros(0, Global_vDisparityParams->totalCols, CV_32FC1);
	//for (float d = 0; d < Global_vDisparityParams->totalCols; d++)
	//{
	//	Zs = (fx * stereoParams.b) / d;
	//	Ys_v0 = (0 + Zs*sin(stereoParams.tilt) - stereoParams.cY) / cos(stereoParams.tilt);
	//	vBase = stereoParams.v0 - fy * (Ys_v0 / Zs);
	//	Global_vDisparityGlobals->vDisparityImage_Baseline.at<float>(0, d) = vBase;
	//}
	//Create real baseline

	if (!totalNumOfLines <= 0) //if there are lines/line
	{
		Global_vDisparityGlobals->allLines[0][3] = Global_vDisparityGlobals->allLines[0][3] + shifted_rowRange.start;
		Global_vDisparityGlobals->allLines[0][1] = Global_vDisparityGlobals->allLines[0][1] + shifted_rowRange.start;
		Global_vDisparityGlobals->allLines[0][2] = Global_vDisparityGlobals->allLines[0][2] + shifted_colRange.start;
		Global_vDisparityGlobals->allLines[0][0] = Global_vDisparityGlobals->allLines[0][0] + shifted_colRange.start;

		x1 = Global_vDisparityGlobals->allLines[0][0]; //first line ([0]) is the most voted line according to the function
		y1 = Global_vDisparityGlobals->allLines[0][1];
		x2 = Global_vDisparityGlobals->allLines[0][2];
		y2 = Global_vDisparityGlobals->allLines[0][3];

		if (!(x2 - x1) == 0)//else line_angle_degree = 90 and thus tangent is undefined
		{
			currentLine_m = ((y2 - y1) / (x2 - x1));
			currentLineAngleRadian = std::atan(currentLine_m);
			currentLineAngleDegree = currentLineAngleRadian*(180 / PI);

			if (currentLineAngleDegree > Global_vDisparityParams->bestLine_minDegreeThreshold && currentLineAngleDegree < Global_vDisparityParams->bestLine_maxDegreeThreshold)
			{
				maxLineIndex = 0;
				isFound = true;
			}
		}

		if (!isFound && totalNumOfLines>0)//if the first line (most voted line) does not match the criterias then check the other lines if they match
		{
			for (size_t i = 1; i < totalNumOfLines; i++)
			{
				Global_vDisparityGlobals->allLines[i][3] = Global_vDisparityGlobals->allLines[i][3] + shifted_rowRange.start;
				Global_vDisparityGlobals->allLines[i][1] = Global_vDisparityGlobals->allLines[i][1] + shifted_rowRange.start;
				Global_vDisparityGlobals->allLines[i][2] = Global_vDisparityGlobals->allLines[i][2] + shifted_colRange.start;
				Global_vDisparityGlobals->allLines[i][0] = Global_vDisparityGlobals->allLines[i][0] + shifted_colRange.start;

				x1 = Global_vDisparityGlobals->allLines[i][0];
				y1 = Global_vDisparityGlobals->allLines[i][1];
				x2 = Global_vDisparityGlobals->allLines[i][2];
				y2 = Global_vDisparityGlobals->allLines[i][3];

				currentLineLength = sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));

				if (!(x2 - x1) == 0)//else line_angle_degree = 90 and thus tangent is undefined
				{
					currentLine_m = ((y2 - y1) / (x2 - x1));
					currentLineAngleRadian = std::atan(currentLine_m);
					currentLineAngleDegree = currentLineAngleRadian*(180 / PI);
					//std::cout << i << ". " << " x1:" << x1 << " y1:" << y1 << " x2:" << x2 << " y2:" << y2 << " len:" << currentLineLength << " m:" << currentLine_m << " rad:" << currentLineAngleRadian << " deg:" << currentLineAngleDegree << endl;

					if (currentLineLength > maxLinelength && (currentLineAngleDegree > Global_vDisparityParams->bestLine_minDegreeThreshold && currentLineAngleDegree < Global_vDisparityParams->bestLine_maxDegreeThreshold)) {
						maxLinelength = currentLineLength;
						maxLineIndex = i;
						isFound = true;
					}
				}
			}

			if (!isFound) maxLineIndex = 0; //if nothing is found just accept the first line
		}
	}
	else//if there are no lines just create a hypotatical line
	{
		cv::Vec4i line;
		line[0] = 0;
		line[1] = 165;
		line[2] = ((d + 1) / 2) - 1;
		line[3] = v;

		Global_vDisparityGlobals->allLines.push_back(line);
		maxLineIndex = 0;
	}

	//v = m*d + b
	Global_vDisparityGlobals->piecewiseBestLine = Global_vDisparityGlobals->allLines[maxLineIndex];
	float max_x1 = Global_vDisparityGlobals->piecewiseBestLine[0];
	float max_y1 = Global_vDisparityGlobals->piecewiseBestLine[1];
	float max_x2 = Global_vDisparityGlobals->piecewiseBestLine[2];
	float max_y2 = Global_vDisparityGlobals->piecewiseBestLine[3];

	Global_vDisparityGlobals->bestRoadLine_m = ((max_y2 - max_y1) / (max_x2 - max_x1));
	Global_vDisparityGlobals->bestRoadLine_b = max_y1 - Global_vDisparityGlobals->bestRoadLine_m * max_x1;
	Global_vDisparityGlobals->bestRoadLine_a = 0;

	float v_intersect = Global_vDisparityGlobals->bestRoadLine_b; //x1 = 0 v = y1 (0,v_intersect)
	float d_intersect = (v - Global_vDisparityGlobals->bestRoadLine_b) / Global_vDisparityGlobals->bestRoadLine_m; //y2 = v (d_intersect,v)

	Global_vDisparityGlobals->bestRoadLine[0] = 0;
	Global_vDisparityGlobals->bestRoadLine[1] = (int)round(v_intersect);
	Global_vDisparityGlobals->bestRoadLine[2] = (int)round(d_intersect);
	Global_vDisparityGlobals->bestRoadLine[3] = v;

	Global_vDisparityGlobals->bestRoadLine_vIntersection = Global_vDisparityGlobals->bestRoadLine[1];//could be a fixed value, now v = b where intersects (Labayrade,2002)
	Global_vDisparityGlobals->bestRoadLine_dInterSection = Global_vDisparityGlobals->bestRoadLine[2];

	cv::Mat colorV;
	cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV, CV_GRAY2BGR);
	cv::line(colorV, Point(0, Global_vDisparityGlobals->bestRoadLine_vIntersection), Point(Global_vDisparityGlobals->bestRoadLine_dInterSection, Global_vDisparityParams->totalRows - 1), cv::Scalar(0, 0, 255));
	cv::line(colorV, Point(Global_vDisparityGlobals->piecewiseBestLine[0], Global_vDisparityGlobals->piecewiseBestLine[1]), Point(Global_vDisparityGlobals->piecewiseBestLine[2], Global_vDisparityGlobals->piecewiseBestLine[3]), cv::Scalar(0, 255, 0));
	//cout << "hey";
}


inline void StereoImageBuffer::set_BestLine_From_VDisparity_Regression() const
{
	clock_t t1, t2;
	float diff_clock;

	cv::Range shifted_rowRange;
	cv::Range shifted_colRange;

	//find the line boundaries and the line equation parameters
	int v_max = Global_vDisparityParams->totalRows - 1;
	int d_max = Global_vDisparityParams->totalCols - 1;

	float maxLinelength = 0;
	float maxLineDegree = 90;
	float currentLineLength, currentLineAngleRadian, currentLineAngleDegree, currentLine_m, currentLine_b;
	float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
	bool isFound = false;
	bool isFound2 = false;
	int maxLineIndex = 0;
	int mostLeftVerticalPointIndex = 0;
	bool ismostLeftVerticalFound = false;
	float mostLeftVerticalPointD = d_max;

	int second_maxLineIndex = 0;
	int rightMostPointIndex = 0;
	float currentRightPointX, currentBottomPointY, currentBottomPointX, currentTopPointY, currentTopPointX;
	float maxRightPoint = 0, maxTopPoint = 0;

	Mat vDisp2 = Global_vDisparityGlobals->vDisparityImage;

	//search in the shifted row and col to get faster
	shifted_rowRange.start = Global_vDisparityParams->shiftedStartRangeV;
	shifted_rowRange.end = Global_vDisparityParams->totalRows;
	shifted_colRange.start = Global_vDisparityParams->shiftedStartRangeD;
	shifted_colRange.end = Global_vDisparityParams->totalCols;

	Global_vDisparityGlobals->vDisparityImage_shifted = Global_vDisparityGlobals->vDisparityImage(shifted_rowRange, shifted_colRange);
	//search in the shifted row and col to get faster		

	cv::GaussianBlur(Global_vDisparityGlobals->vDisparityImage_shifted, Global_vDisparityGlobals->vDisparityImage_blurred, Global_vDisparityParams->gaussianBlur_size, Global_vDisparityParams->gaussianBlur_sigmaX, Global_vDisparityParams->gaussianBlur_sigmaY, Global_vDisparityParams->gaussianBlur_borderType);
	cv::threshold(Global_vDisparityGlobals->vDisparityImage_blurred, Global_vDisparityGlobals->vDisparityImage_thresholded, Global_vDisparityParams->thresholding_threshold, Global_vDisparityParams->thresholding_maxVal, Global_vDisparityParams->thresholding_type);
	cv::HoughLinesP(Global_vDisparityGlobals->vDisparityImage_thresholded, Global_vDisparityGlobals->allLines, Global_vDisparityParams->houghLinesP_rho, Global_vDisparityParams->houghLinesP_theta, Global_vDisparityParams->houghLinesP_threshold, Global_vDisparityParams->houghLinesP_minLineLength, Global_vDisparityParams->houghLinesP_maxLineGap);

	int totalNumOfLines = Global_vDisparityGlobals->allLines.size();

	Mat vDisp = Global_vDisparityGlobals->vDisparityImage;
	Mat vDisp_shift = Global_vDisparityGlobals->vDisparityImage_shifted;
	Mat vDisp_thresholded = Global_vDisparityGlobals->vDisparityImage_thresholded;

	cv::Mat colorV_All_Lines;
	cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV_All_Lines, CV_GRAY2BGR);

	cv::Mat colorV_Inside;
	cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV_Inside, CV_GRAY2BGR);

	cv::Mat colorV_Outside;
	cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV_Outside, CV_GRAY2BGR);

	cv::Mat colorV_Lines;
	cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV_Lines, CV_GRAY2BGR);
	Mat x, y;//for curve points
	Mat V2 = Mat::zeros(Global_vDisparityGlobals->vDisparityImage.size(), Global_vDisparityGlobals->vDisparityImage_thresholded.type());
	Mat V3 = Mat::zeros(cv::Size(Global_vDisparityGlobals->vDisparityImage.cols * 2, Global_vDisparityGlobals->vDisparityImage.rows), Global_vDisparityGlobals->vDisparityImage_thresholded.type());
	cv::Mat colorV;

	////check object bottom lines with calibration parameters instead of vdisp bestline
	//std::vector<float> vDispLinef;
	//std::vector<int> vDispLinei;
	//float bottomYw = 0.0;
	//int out_vi;

	cv::Mat colorV1;
	cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV1, CV_GRAY2BGR);
	//
	//for (float dx = 1; dx < Global_vDisparityParams->totalCols; dx++)
	//{
	//	float v = get_v_From_Calibration(dx, bottomYw, out_vi);
	//	vDispLinef.push_back(v);
	//	vDispLinei.push_back(out_vi);
	//	x1 = dx;
	//	y1 = v;
	//	circle(colorV1, Point(x1, y1), cvRound((double)2 / 2), Scalar(0, 255, 0), -1);
	//}
	////check object bottom lines with calibration parameters instead of vdisp bestline

	if (totalNumOfLines > 0) //if there are lines/line
	{
		if (!isFound && totalNumOfLines>0)//if the first line (most voted line) does not match the criterias then check the other lines if they match
		{
			for (size_t i = 0; i < totalNumOfLines; i++)
			{
				Global_vDisparityGlobals->allLines[i][3] = Global_vDisparityGlobals->allLines[i][3] + shifted_rowRange.start;
				Global_vDisparityGlobals->allLines[i][1] = Global_vDisparityGlobals->allLines[i][1] + shifted_rowRange.start;
				Global_vDisparityGlobals->allLines[i][2] = Global_vDisparityGlobals->allLines[i][2] + shifted_colRange.start;
				Global_vDisparityGlobals->allLines[i][0] = Global_vDisparityGlobals->allLines[i][0] + shifted_colRange.start;

				x1 = Global_vDisparityGlobals->allLines[i][0];
				y1 = Global_vDisparityGlobals->allLines[i][1];
				x2 = Global_vDisparityGlobals->allLines[i][2];
				y2 = Global_vDisparityGlobals->allLines[i][3];


				if ((int)x1 % 3 == 0)
				{
					cv::line(colorV_All_Lines, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 255, 0));
				}
				else
				{
					if ((int)x1 % 3 == 1)
					{
						cv::line(colorV_All_Lines, Point(x1, y1), Point(x2, y2), cv::Scalar(255, 0, 0));
					}
					else
					{
						cv::line(colorV_All_Lines, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 0, 255));
					}
				}


				currentLineLength = sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));

				//if (!(x2 - x1) == 0)//else line_angle_degree = 90 and thus tangent is undefined
				currentLine_m = ((y2 - y1) / (x2 - x1));
				currentLine_b = y1 - currentLine_m * x1;
				currentLineAngleRadian = std::atan(currentLine_m);
				currentLineAngleDegree = currentLineAngleRadian*(180 / PI);
				//std::cout << i << ". " << " x1:" << x1 << " y1:" << y1 << " x2:" << x2 << " y2:" << y2 << " len:" << currentLineLength << " m:" << currentLine_m << " rad:" << currentLineAngleRadian << " deg:" << currentLineAngleDegree << endl;

				if (x1 > x2)
					currentRightPointX = x1;
				else
					currentRightPointX = x2;

				if (y1 > y2)
				{
					currentBottomPointX = x1;
					currentBottomPointY = y1;
					currentTopPointX = x2;
					currentTopPointY = y2;
				}
				else
				{
					currentBottomPointX = x2;
					currentBottomPointY = y2;
					currentTopPointX = x1;
					currentTopPointY = y1;
				}

				if (currentBottomPointX > 2 && abs(currentLineAngleDegree) >= 85 && abs(currentLineAngleDegree) <= Global_vDisparityParams->bestLineCurve_maxDegreeThreshold)//Add only the bottom point
				{
					if (currentBottomPointX < mostLeftVerticalPointD)
					{
						mostLeftVerticalPointD = currentBottomPointX - 2;
						mostLeftVerticalPointIndex = i;
						ismostLeftVerticalFound = true;
					}
				}

				bool degreeFound = false;
				//currentRightPointX since road profile lines are left leaning
				if (currentRightPointX > Global_vDisparityParams->bestLineCurve_leftShiftD && currentBottomPointY > Global_vDisparityParams->horizonLineV && currentLineLength > Global_vDisparityParams->bestLineCurve_minLenLengthThreshold)
				{
					float middle_x1, middle_y1, middle_x2, middle_y2, middle_x3, middle_y3;

					if (((currentLineAngleDegree >= 88 && currentLineAngleDegree <= Global_vDisparityParams->bestLineCurve_maxDegreeThreshold) || currentLineAngleDegree == -90) && currentBottomPointX < (Global_vDisparityParams->totalCols - 5 * Global_vDisparityParams->bestLineCurve_leftShiftD))//Add only the bottom point
					{
						x.push_back(currentBottomPointX);
						y.push_back(currentBottomPointY);
						V2.at<uchar>((int)round(currentBottomPointY), (int)round(currentBottomPointX)) = 255;
						degreeFound = true;

						if ((int)x1 % 3 == 0)
						{
							cv::line(colorV_Inside, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 255, 0));
						}
						else
						{
							if ((int)x1 % 3 == 1)
							{
								cv::line(colorV_Inside, Point(x1, y1), Point(x2, y2), cv::Scalar(255, 0, 0));
							}
							else
							{
								cv::line(colorV_Inside, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 0, 255));
							}
						}
					}

					if ((currentLineAngleDegree >= 86 && currentLineAngleDegree < 88) && currentBottomPointX < (Global_vDisparityParams->totalCols - 4 * Global_vDisparityParams->bestLineCurve_leftShiftD))//Add the bottom point and the point between the bottom and the middle point
					{
						middle_x1 = (currentBottomPointX + currentTopPointX) / 2;

						middle_x2 = (middle_x1 + currentBottomPointX) / 2;
						middle_y2 = currentLine_m*middle_x2 + currentLine_b;

						x.push_back(currentBottomPointX);
						y.push_back(currentBottomPointY);
						V2.at<uchar>((int)round(currentBottomPointY), (int)round(currentBottomPointX)) = 255;

						x.push_back(middle_x2);
						y.push_back(middle_y2);
						V2.at<uchar>((int)round(middle_y2), (int)round(middle_x2)) = 255;

						degreeFound = true;

						if ((int)x1 % 3 == 0)
						{
							cv::line(colorV_Inside, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 255, 0));
						}
						else
						{
							if ((int)x1 % 3 == 1)
							{
								cv::line(colorV_Inside, Point(x1, y1), Point(x2, y2), cv::Scalar(255, 0, 0));
							}
							else
							{
								cv::line(colorV_Inside, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 0, 255));
							}
						}
					}

					if ((currentLineAngleDegree >= 84 && currentLineAngleDegree < 86) && currentBottomPointX < (Global_vDisparityParams->totalCols - 3 * Global_vDisparityParams->bestLineCurve_leftShiftD))//Add the bottom point and the point between the bottom and the middle point
					{
						middle_x1 = (currentBottomPointX + currentTopPointX) / 2;
						middle_y1 = currentLine_m*middle_x1 + currentLine_b;

						middle_x2 = (middle_x1 + currentBottomPointX) / 2;
						middle_y2 = currentLine_m*middle_x2 + currentLine_b;

						x.push_back(currentBottomPointX);
						y.push_back(currentBottomPointY);
						V2.at<uchar>((int)round(currentBottomPointY), (int)round(currentBottomPointX)) = 255;

						x.push_back(middle_x2);
						y.push_back(middle_y2);
						V2.at<uchar>((int)round(middle_y2), (int)round(middle_x2)) = 255;

						x.push_back(middle_x1);
						y.push_back(middle_y1);
						V2.at<uchar>((int)round(middle_y1), (int)round(middle_x1)) = 255;

						degreeFound = true;

						if ((int)x1 % 3 == 0)
						{
							cv::line(colorV_Inside, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 255, 0));
						}
						else
						{
							if ((int)x1 % 3 == 1)
							{
								cv::line(colorV_Inside, Point(x1, y1), Point(x2, y2), cv::Scalar(255, 0, 0));
							}
							else
							{
								cv::line(colorV_Inside, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 0, 255));
							}
						}
					}

					if (currentLineAngleDegree >= 82 && currentLineAngleDegree < 84)
					{
						middle_x1 = (currentBottomPointX + currentTopPointX) / 2;
						middle_y1 = currentLine_m*middle_x1 + currentLine_b;

						middle_x2 = (middle_x1 + currentBottomPointX) / 2;
						middle_y2 = currentLine_m*middle_x2 + currentLine_b;

						middle_x3 = (middle_x1 + currentTopPointX) / 2;
						middle_y3 = currentLine_m*middle_x3 + currentLine_b;

						x.push_back(currentBottomPointX);
						y.push_back(currentBottomPointY);
						V2.at<uchar>((int)round(currentBottomPointY), (int)round(currentBottomPointX)) = 255;

						x.push_back(middle_x2);
						y.push_back(middle_y2);
						V2.at<uchar>((int)round(middle_y2), (int)round(middle_x2)) = 255;

						x.push_back(middle_x1);
						y.push_back(middle_y1);
						V2.at<uchar>((int)round(middle_y1), (int)round(middle_x1)) = 255;

						x.push_back(middle_x3);
						y.push_back(middle_y3);
						V2.at<uchar>((int)round(middle_y3), (int)round(middle_x3)) = 255;

						degreeFound = true;

						if ((int)x1 % 3 == 0)
						{
							cv::line(colorV_Inside, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 255, 0));
						}
						else
						{
							if ((int)x1 % 3 == 1)
							{
								cv::line(colorV_Inside, Point(x1, y1), Point(x2, y2), cv::Scalar(255, 0, 0));
							}
							else
							{
								cv::line(colorV_Inside, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 0, 255));
							}
						}
					}

					if (currentLineAngleDegree >= Global_vDisparityParams->bestLineCurve_minDegreeThreshold && currentLineAngleDegree < 82)//Add the bottom point and the point between the bottom and the middle point
					{
						middle_x1 = (currentBottomPointX + currentTopPointX) / 2;
						middle_y1 = currentLine_m*middle_x1 + currentLine_b;

						middle_x2 = (middle_x1 + currentBottomPointX) / 2;
						middle_y2 = currentLine_m*middle_x2 + currentLine_b;

						middle_x3 = (middle_x1 + currentTopPointX) / 2;
						middle_y3 = currentLine_m*middle_x3 + currentLine_b;

						x.push_back(currentBottomPointX);
						y.push_back(currentBottomPointY);
						V2.at<uchar>((int)round(currentBottomPointY), (int)round(currentBottomPointX)) = 255;

						x.push_back(middle_x2);
						y.push_back(middle_y2);
						V2.at<uchar>((int)round(middle_y2), (int)round(middle_x2)) = 255;

						x.push_back(middle_x1);
						y.push_back(middle_y1);
						V2.at<uchar>((int)round(middle_y1), (int)round(middle_x1)) = 255;

						x.push_back(middle_x3);
						y.push_back(middle_y3);
						V2.at<uchar>((int)round(middle_y3), (int)round(middle_x3)) = 255;

						x.push_back(currentTopPointX);
						y.push_back(currentTopPointY);
						V2.at<uchar>((int)round(currentTopPointY), (int)round(currentTopPointX)) = 255;

						degreeFound = true;

						if ((int)x1 % 3 == 0)
						{
							cv::line(colorV_Inside, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 255, 0));
						}
						else
						{
							if ((int)x1 % 3 == 1)
							{
								cv::line(colorV_Inside, Point(x1, y1), Point(x2, y2), cv::Scalar(255, 0, 0));
							}
							else
							{
								cv::line(colorV_Inside, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 0, 255));
							}
						}
					}

					if (!degreeFound)
					{
						if ((int)x1 % 3 == 0)
						{
							cv::line(colorV_Outside, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 255, 0));
						}
						else
						{
							if ((int)x1 % 3 == 1)
							{
								cv::line(colorV_Outside, Point(x1, y1), Point(x2, y2), cv::Scalar(255, 0, 0));
							}
							else
							{
								cv::line(colorV_Outside, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 0, 255));
							}
						}
					}
				}
				else
				{
					if ((int)x1 % 3 == 0)
					{
						cv::line(colorV_Outside, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 255, 0));
					}
					else
					{
						if ((int)x1 % 3 == 1)
						{
							cv::line(colorV_Outside, Point(x1, y1), Point(x2, y2), cv::Scalar(255, 0, 0));
						}
						else
						{
							cv::line(colorV_Outside, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 0, 255));
						}
					}
				}

				//cout << "";

				////find the peacewise line, it should be the one of the best 10 lines and the smallest degree
				//if ((i < 10) && (maxLineDegree - currentLineAngleDegree > 1) && (currentLineLength > maxLinelength) && (currentLineAngleDegree > Global_vDisparityParams->bestLine_minDegreeThreshold && currentLineAngleDegree < Global_vDisparityParams->bestLine_maxDegreeThreshold) && currentLineLength > Global_vDisparityParams->bestLine_minLenLengthThreshold)
				//{
				//	maxLineDegree = currentLineAngleDegree;
				//	maxLinelength = currentLineLength;
				//	maxLineIndex = i;
				//}

			}// for

			if (x.rows > 0)
			{
				//add the horizon line as well
				x.push_back((float)0);
				y.push_back((float)Global_vDisparityParams->horizonLineV);

				int order = 2;
				Mat fit_weights(order + 1, 1, CV_32FC1);
				cvPolyfit(x, y, fit_weights, order); //find the polynom by regression

				if (order == 2)
				{
					Global_vDisparityGlobals->bestRoadLine_m = fit_weights.at<float>(1, 0);//x
					Global_vDisparityGlobals->bestRoadLine_b = fit_weights.at<float>(0, 0);//b
					Global_vDisparityGlobals->bestRoadLine_a = fit_weights.at<float>(2, 0);//x^2
				}
				else
				{
					if (order == 1)
					{
						Global_vDisparityGlobals->bestRoadLine_m = fit_weights.at<float>(1, 0);//x
						Global_vDisparityGlobals->bestRoadLine_b = fit_weights.at<float>(0, 0);//b

						fit_weights.push_back((float)0.0);
						Global_vDisparityGlobals->bestRoadLine_a = fit_weights.at<float>(2, 0);//x^2
					}
				}

				std::vector<Point2f> curvePoints;

				//cout << "X: " << x.rows << "[ ";
				//for (int i = 0; i < x.rows; i++)
				//{
				//	cout << x.at<float>(i, 0) << " ";
				//}
				//cout << "]" << endl;

				//cout << "Y: " << y.rows << "[ ";
				//for (int i = 0; i < y.rows; i++)
				//{
				//	cout << y.at<float>(i, 0) << " ";
				//}
				//cout << "]" << endl;

				//cout << roundf(fit_weights.at<float>(1, 0) * 1000) / 1000 << "x + " << roundf(fit_weights.at<float>(2, 0) * 1000) / 1000 << "x^2 +" << roundf(fit_weights.at<float>(0, 0) * 1000) / 1000 << endl;
				//cout << roundf(fit_weights.at<float>(1, 0) * 1000) / 1000 << "x + " << roundf(fit_weights.at<float>(2, 0) * 1000) / 1000 << "x^2 +" << roundf(fit_weights.at<float>(3, 0) * 1000) / 1000 << "x^3 +" << roundf(fit_weights.at<float>(0, 0) * 1000) / 1000 << endl;

				int start_point_x = 0;
				int end_point_x = d_max * 2;
				for (float x = start_point_x; x <= end_point_x; x += 1) {
					float y = fit_weights.at<float>(1, 0)*x + fit_weights.at<float>(2, 0)*x*x + fit_weights.at<float>(0, 0);
					//float y = fit_weights.at<float>(1, 0)*x + fit_weights.at<float>(2, 0)*x*x + fit_weights.at<float>(3, 0)*x*x*x + fit_weights.at<float>(0, 0);
					//cout << x << " = " << round(y) << endl;
					Point2f new_point = Point2f(x, y);                  //resized to better visualize
					curvePoints.push_back(new_point);                   //add point to vector/list
				}


				cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV, CV_GRAY2BGR);
				cv::cvtColor(V3, V3, CV_GRAY2BGR);

				//Option 1: use polylines
				Mat curve(curvePoints, true);
				curve.convertTo(curve, CV_32S); //adapt type for polylines
				polylines(colorV, curve, false, Scalar(255), 2, CV_AA);


				//Option 2: use line with each pair of consecutives points
				for (int i = 0; i < curvePoints.size() - 1; i++) {
					if (i <= d_max)
					{
						line(colorV1, curvePoints[i], curvePoints[i + 1], Scalar(0, 0, 255), 2, CV_AA);
						circle(colorV1, curvePoints[i], cvRound((double)2 / 2), Scalar(0, 255, 0), -1);

						line(V3, curvePoints[i], curvePoints[i + 1], Scalar(0, 0, 255), 2, CV_AA);
						circle(V3, curvePoints[i], cvRound((double)2 / 2), Scalar(0, 255, 0), -1);
					}
					else
					{
						line(V3, curvePoints[i], curvePoints[i + 1], Scalar(0, 0, 255), 2, CV_AA);
						circle(V3, curvePoints[i], cvRound((double)2 / 2), Scalar(0, 255, 0), -1);
					}

				}

				//int v_check = get_v_From_VDisparity_RoadProfile(49);
				//int d_check = get_d_From_VDisparity_RoadProfile(v_check);

				if (!ismostLeftVerticalFound)
					mostLeftVerticalPointD = 2;

				polylines(V2, curve, false, Scalar(255), 2, CV_AA);
				cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV, CV_GRAY2BGR);
				cv::line(colorV, Point(Global_vDisparityGlobals->bestRoadLine[0], Global_vDisparityGlobals->bestRoadLine[1]), Point(Global_vDisparityGlobals->bestRoadLine[2], Global_vDisparityGlobals->bestRoadLine[3]), cv::Scalar(0, 0, 255));
				cv::line(colorV, Point(Global_vDisparityGlobals->piecewiseBestLine[0], Global_vDisparityGlobals->piecewiseBestLine[1]), Point(Global_vDisparityGlobals->piecewiseBestLine[2], Global_vDisparityGlobals->piecewiseBestLine[3]), cv::Scalar(0, 255, 0));
				cv::line(colorV, Point(Global_vDisparityGlobals->allLines[mostLeftVerticalPointIndex][0], Global_vDisparityGlobals->allLines[mostLeftVerticalPointIndex][1]), Point(Global_vDisparityGlobals->allLines[mostLeftVerticalPointIndex][2], Global_vDisparityGlobals->allLines[mostLeftVerticalPointIndex][3]), cv::Scalar(255, 0, 0));
				//cout << endl;
			}
			else//if there are no curve points either create a hypotatical line
			{
				cv::Vec4i line;
				line[0] = 0;
				line[1] = Global_vDisparityParams->horizonLineV;
				line[2] = d_max - 2;
				line[3] = v_max;

				Global_vDisparityGlobals->allLines.push_back(line);
				maxLineIndex = Global_vDisparityGlobals->allLines.size() - 1;
			}
		}
	}
	else//if there are no lines just create a hypotatical line
	{
		cv::Vec4i line;
		line[0] = 0;
		line[1] = Global_vDisparityParams->horizonLineV;
		line[2] = d_max - 2;
		line[3] = v_max;

		Global_vDisparityGlobals->allLines.push_back(line);
		maxLineIndex = 0;
		second_maxLineIndex = 0;
	}

	//assign the piecewise and the start and end point of the road
	//Global_vDisparityGlobals->piecewiseBestLine = Global_vDisparityGlobals->allLines[maxLineIndex];

	//int piecewise_x1 = (int)round(Global_vDisparityGlobals->piecewiseBestLine[0]);
	//int piecewise_x2 = (int)round(Global_vDisparityGlobals->piecewiseBestLine[2]);
	//int piecewise_y1 = get_v_From_VDisparity_RoadProfile(piecewise_x1);
	//int piecewise_y2 = get_v_From_VDisparity_RoadProfile(piecewise_x2);

	if (!ismostLeftVerticalFound)
		mostLeftVerticalPointD = 2;

	int piecewise_x1 = mostLeftVerticalPointD;
	int piecewise_x2 = d_max;
	int piecewise_y1 = get_v_From_VDisparity_RoadProfile(piecewise_x1);
	int piecewise_y2 = get_v_From_VDisparity_RoadProfile(piecewise_x2);

	Global_vDisparityGlobals->piecewiseBestLine = Vec4i(piecewise_x1, piecewise_y1, piecewise_x2, piecewise_y2);

	//if (piecewise_x1 < piecewise_x2)
	//{
	Global_vDisparityGlobals->piecewiseBestLine_RoadEndPoint = Point2i(piecewise_x1, piecewise_y1);
	Global_vDisparityGlobals->piecewiseBestLine_RoadStartPoint = Point2i(piecewise_x2, piecewise_y2);
	//}
	//else
	//{
	//	Global_vDisparityGlobals->piecewiseBestLine_RoadEndPoint = Point2i(piecewise_x2, piecewise_y2);
	//	Global_vDisparityGlobals->piecewiseBestLine_RoadStartPoint = Point2i(piecewise_x1, piecewise_y1);
	//}

	int v_intersect = get_v_From_VDisparity_RoadProfile(0);
	int d_intersect = get_d_From_VDisparity_RoadProfile(v_max);


	Global_vDisparityGlobals->bestRoadLine[0] = 0;
	Global_vDisparityGlobals->bestRoadLine[1] = v_intersect;
	Global_vDisparityGlobals->bestRoadLine[2] = d_intersect;
	if (d_intersect < d_max)
		Global_vDisparityGlobals->bestRoadLine[3] = v_max;
	else
		Global_vDisparityGlobals->bestRoadLine[3] = get_v_From_VDisparity_RoadProfile(d_max);

	Global_vDisparityGlobals->bestRoadLine_vIntersection = Global_vDisparityGlobals->bestRoadLine[1];//could be a fixed value, now v = b where intersects (Labayrade,2002)
	Global_vDisparityGlobals->bestRoadLine_dInterSection = Global_vDisparityGlobals->bestRoadLine[2];


	if (colorV.channels() == 1)
		cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV, CV_GRAY2BGR);

	std::vector<Point2f> curvePoints;
	int start_point_x = 0;
	int end_point_x = d_max;
	for (float x = start_point_x; x <= end_point_x; x += 1) {
		float y = Global_vDisparityGlobals->bestRoadLine_m*x + Global_vDisparityGlobals->bestRoadLine_a*x*x + Global_vDisparityGlobals->bestRoadLine_b;
		//float y = fit_weights.at<float>(1, 0)*x + fit_weights.at<float>(2, 0)*x*x + fit_weights.at<float>(3, 0)*x*x*x + fit_weights.at<float>(0, 0);
		//cout << x << " = " << round(y) << endl;
		Point2f new_point = Point2f(x, y);                  //resized to better visualize
		curvePoints.push_back(new_point);                   //add point to vector/list
	}

	Mat curve(curvePoints, true);
	curve.convertTo(curve, CV_32S); //adapt type for polylines
	polylines(colorV, curve, false, Scalar(255), 2, CV_AA);

	cv::line(colorV, Point(Global_vDisparityGlobals->bestRoadLine[0], Global_vDisparityGlobals->bestRoadLine[1]), Point(Global_vDisparityGlobals->bestRoadLine[2], Global_vDisparityGlobals->bestRoadLine[3]), cv::Scalar(0, 0, 255));
	cv::line(colorV, Point(Global_vDisparityGlobals->piecewiseBestLine[0], Global_vDisparityGlobals->piecewiseBestLine[1]), Point(Global_vDisparityGlobals->piecewiseBestLine[2], Global_vDisparityGlobals->piecewiseBestLine[3]), cv::Scalar(0, 255, 0));
	//cout << endl;


}

inline void StereoImageBuffer::set_BestLine_From_VDisparity_Curve() const
{
	cv::Range shifted_rowRange;
	cv::Range shifted_colRange;

	//find the line boundaries and the line equation parameters
	int v_max = Global_vDisparityParams->totalRows - 1;
	int d_max = Global_vDisparityParams->totalCols - 1;

	float maxLinelength = 0;
	float currentLineLength, currentLineAngleRadian, currentLineAngleDegree, currentLine_m, currentLine_b;
	float x1, y1, x2, y2;
	bool isFound = false;
	bool isFound2 = false;
	int maxLineIndex = 0;

	int second_maxLineIndex = 0;
	int rightMostPointIndex = 0;
	float currentRightPointX, currentBottomPointY, currentBottomPointX, currentTopPointY, currentTopPointX;
	float maxRightPoint = 0, maxTopPoint = 0;

	//search in the shifted row and col to get faster
	shifted_rowRange.start = Global_vDisparityParams->shiftedStartRangeV;
	shifted_rowRange.end = Global_vDisparityParams->totalRows;
	shifted_colRange.start = Global_vDisparityParams->shiftedStartRangeD;
	shifted_colRange.end = Global_vDisparityParams->totalCols;

	Global_vDisparityGlobals->vDisparityImage_shifted = Global_vDisparityGlobals->vDisparityImage(shifted_rowRange, shifted_colRange);
	//search in the shifted row and col to get faster		

	cv::GaussianBlur(Global_vDisparityGlobals->vDisparityImage_shifted, Global_vDisparityGlobals->vDisparityImage_blurred, Global_vDisparityParams->gaussianBlur_size, Global_vDisparityParams->gaussianBlur_sigmaX, Global_vDisparityParams->gaussianBlur_sigmaY, Global_vDisparityParams->gaussianBlur_borderType);
	cv::threshold(Global_vDisparityGlobals->vDisparityImage_blurred, Global_vDisparityGlobals->vDisparityImage_thresholded, Global_vDisparityParams->thresholding_threshold, Global_vDisparityParams->thresholding_maxVal, Global_vDisparityParams->thresholding_type);
	cv::HoughLinesP(Global_vDisparityGlobals->vDisparityImage_thresholded, Global_vDisparityGlobals->allLines, Global_vDisparityParams->houghLinesP_rho, Global_vDisparityParams->houghLinesP_theta, Global_vDisparityParams->houghLinesP_threshold, Global_vDisparityParams->houghLinesP_minLineLength, Global_vDisparityParams->houghLinesP_maxLineGap);

	int totalNumOfLines = Global_vDisparityGlobals->allLines.size();

	Mat vDisp = Global_vDisparityGlobals->vDisparityImage;
	Mat vDisp_shift = Global_vDisparityGlobals->vDisparityImage_shifted;
	Mat vDisp_thresholded = Global_vDisparityGlobals->vDisparityImage_thresholded;

	cv::Mat colorV_All_Lines;
	cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV_All_Lines, CV_GRAY2BGR);

	cv::Mat colorV_Lines;
	cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV_Lines, CV_GRAY2BGR);
	Mat x, y;//for curve points
	Mat V2 = Mat::zeros(Global_vDisparityGlobals->vDisparityImage.size(), Global_vDisparityGlobals->vDisparityImage_thresholded.type());
	Mat V3 = Mat::zeros(cv::Size(Global_vDisparityGlobals->vDisparityImage.cols * 2, Global_vDisparityGlobals->vDisparityImage.rows), Global_vDisparityGlobals->vDisparityImage_thresholded.type());
	cv::Mat colorV;

	if (totalNumOfLines > 0) //if there are lines/line
	{
		if (!isFound && totalNumOfLines>0)//if the first line (most voted line) does not match the criterias then check the other lines if they match
		{
			for (size_t i = 0; i < totalNumOfLines; i++)
			{
				Global_vDisparityGlobals->allLines[i][3] = Global_vDisparityGlobals->allLines[i][3] + shifted_rowRange.start;
				Global_vDisparityGlobals->allLines[i][1] = Global_vDisparityGlobals->allLines[i][1] + shifted_rowRange.start;
				Global_vDisparityGlobals->allLines[i][2] = Global_vDisparityGlobals->allLines[i][2] + shifted_colRange.start;
				Global_vDisparityGlobals->allLines[i][0] = Global_vDisparityGlobals->allLines[i][0] + shifted_colRange.start;

				x1 = Global_vDisparityGlobals->allLines[i][0];
				y1 = Global_vDisparityGlobals->allLines[i][1];
				x2 = Global_vDisparityGlobals->allLines[i][2];
				y2 = Global_vDisparityGlobals->allLines[i][3];


				if ((int)x1 % 3 == 0)
				{
					cv::line(colorV_All_Lines, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 255, 0));
				}
				else
				{
					if ((int)x1 % 3 == 1)
					{
						cv::line(colorV_All_Lines, Point(x1, y1), Point(x2, y2), cv::Scalar(255, 0, 0));
					}
					else
					{
						cv::line(colorV_All_Lines, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 0, 255));
					}
				}


				currentLineLength = sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));

				//if (!(x2 - x1) == 0)//else line_angle_degree = 90 and thus tangent is undefined
				currentLine_m = ((y2 - y1) / (x2 - x1));
				currentLine_b = y1 - currentLine_m * x1;
				currentLineAngleRadian = std::atan(currentLine_m);
				currentLineAngleDegree = currentLineAngleRadian*(180 / PI);
				//std::cout << i << ". " << " x1:" << x1 << " y1:" << y1 << " x2:" << x2 << " y2:" << y2 << " len:" << currentLineLength << " m:" << currentLine_m << " rad:" << currentLineAngleRadian << " deg:" << currentLineAngleDegree << endl;

				if (x1 > x2)
					currentRightPointX = x1;
				else
					currentRightPointX = x2;

				if (y1 > y2)
				{
					currentBottomPointX = x1;
					currentBottomPointY = y1;
					currentTopPointX = x2;
					currentTopPointY = y2;
				}
				else
				{
					currentBottomPointX = x2;
					currentBottomPointY = y2;
					currentTopPointX = x1;
					currentTopPointY = y1;
				}

				//currentRightPointX since road profile lines are left leaning, right point should be also the bottom line (45 < degree < 90)
				if ((currentRightPointX > Global_vDisparityParams->bestLineCurve_leftShiftD && currentBottomPointY > Global_vDisparityParams->bestLineCurve_bottomShiftV) && (currentLineAngleDegree > Global_vDisparityParams->bestLineCurve_minDegreeThreshold && currentLineAngleDegree < Global_vDisparityParams->bestLineCurve_maxDegreeThreshold))
				{
					//x.push_back(currentBottomPointX);
					//y.push_back(currentBottomPointY);

					//add middle point 1
					float middle_x1 = (currentBottomPointX + currentTopPointX) / 2;
					float  middle_y1 = currentLine_m*middle_x1 + currentLine_b;

					x.push_back(middle_x1);
					y.push_back(middle_y1);

					//add middle point 1
					float middle_x2 = (middle_x1 + currentBottomPointX) / 2;
					float middle_y2 = currentLine_m*middle_x2 + currentLine_b;

					x.push_back(middle_x2);
					y.push_back(middle_y2);

					//x.push_back(currentTopPointX);//I discard this since bottom is a better roadline profile
					//y.push_back(currentTopPointY);

					//V2.at<uchar>((int)round(y1), (int)round(x1)) = 255;
					//V2.at<uchar>((int)y2, (int)x2) = 255; //I discard this since bottom is a better roadline profile
					V2.at<uchar>((int)round(middle_y1), (int)round(middle_x1)) = 255;
					V2.at<uchar>((int)round(middle_y2), (int)round(middle_x2)) = 255;
				}

				if (currentLineLength > maxLinelength && (currentLineAngleDegree > Global_vDisparityParams->bestLine_minDegreeThreshold && currentLineAngleDegree < Global_vDisparityParams->bestLine_maxDegreeThreshold) && currentLineLength > Global_vDisparityParams->bestLine_minLenLengthThreshold) // if the first ranked line is not suitable, just select the second best scored line
				{
					maxLinelength = currentLineLength;
					maxLineIndex = i;
					isFound = true;

					if (i > 5)//first lines are the best lines according to houghLines
					{
						break;
					}
				}

			}// for

			if (!isFound)// then try polynom equation regression
			{
				if (x.rows > 0)
				{
					//add the horizon line as well
					x.push_back((float)0);
					y.push_back((float)Global_vDisparityParams->horizonLineV);

					int order = 2;
					Mat fit_weights(order + 1, 1, CV_32FC1);
					cvPolyfit(x, y, fit_weights, order); //find the polynom by regression

					Global_vDisparityGlobals->bestRoadLine_m = fit_weights.at<float>(1, 0);//x
					Global_vDisparityGlobals->bestRoadLine_b = fit_weights.at<float>(0, 0);//b
					Global_vDisparityGlobals->bestRoadLine_a = fit_weights.at<float>(2, 0);//x^2

					std::vector<Point2f> curvePoints;

					//cout << "X: " << x.rows << "[ ";
					//for (int i = 0; i < x.rows; i++)
					//{
					//	cout << x.at<float>(i, 0) << " ";
					//}
					//cout << "]" << endl;

					//cout << "Y: " << y.rows << "[ ";
					//for (int i = 0; i < y.rows; i++)
					//{
					//	cout << y.at<float>(i, 0) << " ";
					//}
					//cout << "]" << endl;

					//cout << roundf(fit_weights.at<float>(1, 0) * 1000) / 1000 << "x + " << roundf(fit_weights.at<float>(2, 0) * 1000) / 1000 << "x^2 +" << roundf(fit_weights.at<float>(0, 0) * 1000) / 1000 << endl;
					//cout << roundf(fit_weights.at<float>(1, 0) * 1000) / 1000 << "x + " << roundf(fit_weights.at<float>(2, 0) * 1000) / 1000 << "x^2 +" << roundf(fit_weights.at<float>(3, 0) * 1000) / 1000 << "x^3 +" << roundf(fit_weights.at<float>(0, 0) * 1000) / 1000 << endl;

					int start_point_x = 0;
					int end_point_x = d_max * 2;
					for (float x = start_point_x; x <= end_point_x; x += 1) {
						float y = fit_weights.at<float>(1, 0)*x + fit_weights.at<float>(2, 0)*x*x + fit_weights.at<float>(0, 0);
						//float y = fit_weights.at<float>(1, 0)*x + fit_weights.at<float>(2, 0)*x*x + fit_weights.at<float>(3, 0)*x*x*x + fit_weights.at<float>(0, 0);
						//cout << x << " = " << round(y) << endl;
						Point2f new_point = Point2f(x, y);                  //resized to better visualize
						curvePoints.push_back(new_point);                   //add point to vector/list
					}

					cv::Mat colorV1;
					cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV1, CV_GRAY2BGR);
					cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV, CV_GRAY2BGR);
					cv::cvtColor(V3, V3, CV_GRAY2BGR);

					//Option 1: use polylines
					Mat curve(curvePoints, true);
					curve.convertTo(curve, CV_32S); //adapt type for polylines
					polylines(colorV, curve, false, Scalar(255), 2, CV_AA);
					polylines(V2, curve, false, Scalar(255), 2, CV_AA);

					//Option 2: use line with each pair of consecutives points
					for (int i = 0; i < curvePoints.size() - 1; i++) {
						if (i <= d_max)
						{
							line(colorV1, curvePoints[i], curvePoints[i + 1], Scalar(0, 0, 255), 2, CV_AA);
							circle(colorV1, curvePoints[i], cvRound((double)2 / 2), Scalar(0, 255, 0), -1);

							line(V3, curvePoints[i], curvePoints[i + 1], Scalar(0, 0, 255), 2, CV_AA);
							circle(V3, curvePoints[i], cvRound((double)2 / 2), Scalar(0, 255, 0), -1);
						}
						else
						{
							line(V3, curvePoints[i], curvePoints[i + 1], Scalar(0, 0, 255), 2, CV_AA);
							circle(V3, curvePoints[i], cvRound((double)2 / 2), Scalar(0, 255, 0), -1);
						}

					}

					//int v_check = get_v_From_VDisparity_RoadProfile(49);
					//int d_check = get_d_From_VDisparity_RoadProfile(v_check);

					cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV, CV_GRAY2BGR);
					cv::line(colorV, Point(Global_vDisparityGlobals->bestRoadLine[0], Global_vDisparityGlobals->bestRoadLine[1]), Point(Global_vDisparityGlobals->bestRoadLine[2], Global_vDisparityGlobals->bestRoadLine[3]), cv::Scalar(0, 0, 255));
					cv::line(colorV, Point(Global_vDisparityGlobals->piecewiseBestLine[0], Global_vDisparityGlobals->piecewiseBestLine[1]), Point(Global_vDisparityGlobals->piecewiseBestLine[2], Global_vDisparityGlobals->piecewiseBestLine[3]), cv::Scalar(0, 255, 0));
				}
				else//if there are no curve points either create a hypotatical line
				{
					cv::Vec4i line;
					line[0] = 0;
					line[1] = Global_vDisparityParams->horizonLineV;
					line[2] = d_max - 2;
					line[3] = v_max;

					Global_vDisparityGlobals->allLines.push_back(line);
					maxLineIndex = Global_vDisparityGlobals->allLines.size() - 1;
				}
			}
		}
	}
	else//if there are no lines just create a hypotatical line
	{
		cv::Vec4i line;
		line[0] = 0;
		line[1] = Global_vDisparityParams->horizonLineV;
		line[2] = d_max - 2;
		line[3] = v_max;

		Global_vDisparityGlobals->allLines.push_back(line);
		maxLineIndex = 0;
		second_maxLineIndex = 0;
	}

	//v = m*d + b

	Global_vDisparityGlobals->piecewiseBestLine = Global_vDisparityGlobals->allLines[maxLineIndex];

	Vec4i bestLine = Global_vDisparityGlobals->allLines[maxLineIndex];
	float max_x1 = Global_vDisparityGlobals->piecewiseBestLine[0];
	float max_y1 = Global_vDisparityGlobals->piecewiseBestLine[1];
	float max_x2 = Global_vDisparityGlobals->piecewiseBestLine[2];
	float max_y2 = Global_vDisparityGlobals->piecewiseBestLine[3];

	Mat V2_disp = Global_vDisparityGlobals->vDisparityImage;
	cv::cvtColor(V2_disp, V2_disp, CV_GRAY2BGR);

	ostringstream text_curve, text_line;

	if (x.rows > 0)
	{
		cv::cvtColor(V2, V2, CV_GRAY2BGR);
		std::vector<Point2f> curvePoints;

		//add the horizon line as well
		x.push_back((float)0);
		y.push_back((float)Global_vDisparityParams->horizonLineV);

		int order = 1;
		Mat fit_weights(order + 1, 1, CV_32FC1);

		cvPolyfit(x, y, fit_weights, order);

		//cout << "X: " << x.rows << "[ ";
		//for (int i = 0; i < x.rows; i++)
		//{
		//	cout << x.at<float>(i, 0) << " ";
		//}
		//cout << "]" << endl;

		//cout << "Y: " << y.rows << "[ ";
		//for (int i = 0; i < y.rows; i++)
		//{
		//	cout << y.at<float>(i, 0) << " ";
		//}
		//cout << "]" << endl;

		//cout << roundf(fit_weights.at<float>(1, 0) * 1000) / 1000 << "x + " << roundf(fit_weights.at<float>(2, 0) * 1000) / 1000 << "x^2 +" << roundf(fit_weights.at<float>(0, 0) * 1000) / 1000 << endl;
		//cout << roundf(fit_weights.at<float>(1, 0) * 1000) / 1000 << "x + " << roundf(fit_weights.at<float>(2, 0) * 1000) / 1000 << "x^2 +" << roundf(fit_weights.at<float>(3, 0) * 1000) / 1000 << "x^3 +" << roundf(fit_weights.at<float>(0, 0) * 1000) / 1000 << endl;

		int start_point_x = 0;
		int end_point_x = 63;
		for (float x = start_point_x; x <= end_point_x; x += 1) {
			float y = fit_weights.at<float>(1, 0)*x /*+ fit_weights.at<float>(2, 0)*x*x*/ + fit_weights.at<float>(0, 0);
			//float y = fit_weights.at<float>(1, 0)*x + fit_weights.at<float>(2, 0)*x*x + fit_weights.at<float>(3, 0)*x*x*x + fit_weights.at<float>(0, 0);
			//cout << x << " = " << round(y) << endl;
			Point2f new_point = Point2f(x, y);                  //resized to better visualize
			curvePoints.push_back(new_point);                       //add point to vector/list
		}

		cv::Mat colorV1;
		cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV1, CV_GRAY2BGR);

		//Option 1: use polylines
		Mat curve(curvePoints, true);
		curve.convertTo(curve, CV_32S); //adapt type for polylines
		polylines(colorV1, curve, false, Scalar(255), 2, CV_AA);
		polylines(V2, curve, false, Scalar(255), 2, CV_AA);
		polylines(V2_disp, curve, false, Scalar(255), 2, CV_AA);



		//Option 2: use line with each pair of consecutives points
		for (int i = 0; i < curvePoints.size() - 1; i++) {
			line(colorV1, curvePoints[i], curvePoints[i + 1], Scalar(0, 0, 255), 2, CV_AA);
			circle(colorV1, curvePoints[i], cvRound((double)2 / 2), Scalar(0, 255, 0), -1);
		}


		//text_curve << roundf(fit_weights.at<float>(1, 0) * 1000) / 1000 << "x + " << /*roundf(fit_weights.at<float>(2, 0) * 1000) / 1000 << "x^2 +" <<*/ roundf(fit_weights.at<float>(0, 0) * 1000) / 1000 << endl;
		//cout << "hey";
	}


	int v_intersect, d_intersect;
	if (Global_vDisparityGlobals->bestRoadLine_a == 0) // linear equation
	{
		Global_vDisparityGlobals->bestRoadLine_m = ((max_y2 - max_y1) / (max_x2 - max_x1));
		Global_vDisparityGlobals->bestRoadLine_b = max_y1 - Global_vDisparityGlobals->bestRoadLine_m * max_x1;

		v_intersect = get_v_From_VDisparity_RoadProfile(0); //x1 = 0 v = y1 (0,v_intersect)
		d_intersect = get_d_From_VDisparity_RoadProfile(v_max); //y2 = v (d_intersect,v)

	}
	else // if it is a polynom
	{
		v_intersect = get_v_From_VDisparity_RoadProfile(0);
		d_intersect = get_d_From_VDisparity_RoadProfile(v_max);
	}// else if it is a polynom

	Global_vDisparityGlobals->bestRoadLine[0] = 0;
	Global_vDisparityGlobals->bestRoadLine[1] = v_intersect;
	Global_vDisparityGlobals->bestRoadLine[2] = d_intersect;
	if (d_intersect < d_max)
		Global_vDisparityGlobals->bestRoadLine[3] = v_max;
	else
		Global_vDisparityGlobals->bestRoadLine[3] = get_v_From_VDisparity_RoadProfile(d_max);

	Global_vDisparityGlobals->bestRoadLine_vIntersection = Global_vDisparityGlobals->bestRoadLine[1];//could be a fixed value, now v = b where intersects (Labayrade,2002)
	Global_vDisparityGlobals->bestRoadLine_dInterSection = Global_vDisparityGlobals->bestRoadLine[2];



	//text_line << roundf(Global_vDisparityGlobals->bestRoadLine_m * 1000) / 1000 << "x + " << roundf(Global_vDisparityGlobals->bestRoadLine_b * 1000) / 1000 << endl;

	//cout << text_curve.str();
	//cout << text_line.str() << endl;

	//if (isFound)
	if (colorV.channels() == 1)
		cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV, CV_GRAY2BGR);

	cv::line(colorV, Point(Global_vDisparityGlobals->bestRoadLine[0], Global_vDisparityGlobals->bestRoadLine[1]), Point(Global_vDisparityGlobals->bestRoadLine[2], Global_vDisparityGlobals->bestRoadLine[3]), cv::Scalar(0, 0, 255));
	cv::line(colorV, Point(Global_vDisparityGlobals->piecewiseBestLine[0], Global_vDisparityGlobals->piecewiseBestLine[1]), Point(Global_vDisparityGlobals->piecewiseBestLine[2], Global_vDisparityGlobals->piecewiseBestLine[3]), cv::Scalar(0, 255, 0));
	cv::line(V2_disp, Point(Global_vDisparityGlobals->bestRoadLine[0], Global_vDisparityGlobals->bestRoadLine[1]), Point(Global_vDisparityGlobals->bestRoadLine[2], Global_vDisparityGlobals->bestRoadLine[3]), cv::Scalar(0, 0, 255));
	//cv::line(colorV, Point(bestRightLine[0], bestRightLine[1]), Point(bestRightLine[2], bestRightLine[3]), cv::Scalar(255, 0, 255));
	//cv::line(colorV, Point(secondBestLine[0], secondBestLine[1]), Point(secondBestLine[2], secondBestLine[3]), cv::Scalar(255, 0, 0));
	//cout << "hey" << endl;


}

inline void StereoImageBuffer::set_ImageInfinityPoint_And_VehicleRoadLine(int infinityPointX, int infinityPointY, int roadLineBottomX, int roadLineBottomY)
{
	Global_dDisparityGlobals->infinityPointX = infinityPointX;//the camera parameters could be used to make it automatic !!!!!!!!!
	Global_dDisparityGlobals->infinityPointY = infinityPointY;

	roadLineBottomX = roadLineBottomX - 1;
	roadLineBottomY = roadLineBottomY - 1;

	Global_dDisparityGlobals->vehicleRoadLineRight_m = (static_cast<float>(infinityPointY) - static_cast<float>(roadLineBottomY)) / (static_cast<float>(infinityPointX) - static_cast<float>(roadLineBottomX));
	Global_dDisparityGlobals->vehicleRoadLineRight_b = infinityPointY - Global_dDisparityGlobals->vehicleRoadLineRight_m * infinityPointX;

	Global_dDisparityGlobals->vehicleRoadLineLeft_m = -Global_dDisparityGlobals->vehicleRoadLineRight_m;
	Global_dDisparityGlobals->vehicleRoadLineLeft_b = roadLineBottomY;
}

inline void StereoImageBuffer::set_LabelsOf_UDisparityLines_Using_LabelingAlg()
{
	bool out_onceOccupied = false;
	int out_previousRightMostOccupiedCellIndex = -1;
	int out_labelNum = 0;
	bool isPreviousMaskOccupied = false;
	int d = 0, u = 0;
	float quarterNum;


	//std::vector<MyObject, NAlloc<MyObject>> myGlobal_currentObjectList(Global_maxNumOfObjectAllowed);
#pragma region Udisp Preparation
	//cv::GaussianBlur(Global_uDisparityGlobals->uDisparityImage, Global_uDisparityGlobals->uDisparityImage_blurred, Global_uDisparityParams->gaussianBlur_size, Global_uDisparityParams->gaussianBlur_sigmaX, Global_uDisparityParams->gaussianBlur_sigmaY, Global_uDisparityParams->gaussianBlur_borderType);

	Global_labeledObjectList.clear();
	Global_labeledObjectList.resize(Global_maxNumOfObjectAllowed);

	//if (Global_uDisparityParams->shiftedStartRangeD > 0)
	//{
	cv::Range shifted_rowRange;
	cv::Range shifted_colRange;

	//use shiftedD to ignore low d rows
	shifted_rowRange.start = Global_uDisparityParams->shiftedStartRangeD;
	shifted_rowRange.end = Global_uDisparityParams->totalRows;
	shifted_colRange.start = 0; // no shift (if shift then use Global_vDisparityParams->shiftedStartRangeV )
	shifted_colRange.end = Global_uDisparityParams->totalCols;

	Mat  uDisparityImage, uDisparityImage_thresholded1, uDisparityImage_thresholded2;

	Global_uDisparityGlobals->uDisparityImage_shifted = Global_uDisparityGlobals->uDisparityImage(shifted_rowRange, shifted_colRange);

	cv::Range new_rowRange = shifted_rowRange;

	new_rowRange.start = shifted_rowRange.start;
	new_rowRange.end = Global_uDisparityParams->zeroD;
	Mat uDisparityImage_zeroD = Global_uDisparityGlobals->uDisparityImage(new_rowRange, shifted_colRange);
	Mat uDisparityImage_zeroD_thresholded;
	cv::adaptiveThreshold(uDisparityImage_zeroD, uDisparityImage_zeroD_thresholded, Global_uDisparityParams->thresholding_maxVal, Global_uDisparityParams->thresholding_adaptiveMethod, Global_uDisparityParams->thresholding_type, Global_uDisparityParams->thresholding_blockSize + 2, Global_uDisparityParams->thresholding_SubtractionConstant_C + 4);
	//cv::threshold(uDisparityImage_zeroD, uDisparityImage_zeroD_thresholded, Global_vDisparityParams->thresholding_threshold, Global_vDisparityParams->thresholding_maxVal, Global_vDisparityParams->thresholding_type);

	new_rowRange.start = Global_uDisparityParams->zeroD;
	new_rowRange.end = Global_uDisparityParams->firstQuarterD;// Global_uDisparityParams->secondQuarterD - 1;
	Mat uDisparityImage_firstQuarterD = Global_uDisparityGlobals->uDisparityImage(new_rowRange, shifted_colRange);
	Mat uDisparityImage_firstQuarterD_thresholded;
	cv::adaptiveThreshold(uDisparityImage_firstQuarterD, uDisparityImage_firstQuarterD_thresholded, Global_uDisparityParams->thresholding_maxVal, Global_uDisparityParams->thresholding_adaptiveMethod, Global_uDisparityParams->thresholding_type, Global_uDisparityParams->thresholding_blockSize, Global_uDisparityParams->thresholding_SubtractionConstant_C);
	//cv::threshold(uDisparityImage_firstQuarterD, uDisparityImage_firstQuarterD_thresholded, Global_vDisparityParams->thresholding_threshold, Global_vDisparityParams->thresholding_maxVal, Global_vDisparityParams->thresholding_type);

	new_rowRange.start = Global_uDisparityParams->firstQuarterD;
	new_rowRange.end = Global_uDisparityParams->secondQuarterD;
	Mat uDisparityImage_secondQuarterD = Global_uDisparityGlobals->uDisparityImage(new_rowRange, shifted_colRange);
	Mat uDisparityImage_secondQuarterD_thresholded;
	cv::adaptiveThreshold(uDisparityImage_secondQuarterD, uDisparityImage_secondQuarterD_thresholded, Global_uDisparityParams->thresholding_maxVal, Global_uDisparityParams->thresholding_adaptiveMethod, Global_uDisparityParams->thresholding_type, Global_uDisparityParams->thresholding_blockSize - 4, Global_uDisparityParams->thresholding_SubtractionConstant_C - 10);
	//cv::threshold(uDisparityImage_secondQuarterD, uDisparityImage_secondQuarterD_thresholded, Global_vDisparityParams->thresholding_threshold, Global_vDisparityParams->thresholding_maxVal, Global_vDisparityParams->thresholding_type);

	new_rowRange.start = Global_uDisparityParams->secondQuarterD;
	new_rowRange.end = Global_uDisparityParams->thirdQuarterD;
	Mat uDisparityImage_thirdQuarterD = Global_uDisparityGlobals->uDisparityImage(new_rowRange, shifted_colRange);
	Mat uDisparityImage_thirdQuarterD_thresholded;
	cv::adaptiveThreshold(uDisparityImage_thirdQuarterD, uDisparityImage_thirdQuarterD_thresholded, Global_uDisparityParams->thresholding_maxVal, Global_uDisparityParams->thresholding_adaptiveMethod, Global_uDisparityParams->thresholding_type, Global_uDisparityParams->thresholding_blockSize - 4, Global_uDisparityParams->thresholding_SubtractionConstant_C - 14);
	//cv::threshold(uDisparityImage_thirdQuarterD, uDisparityImage_thirdQuarterD_thresholded, Global_vDisparityParams->thresholding_threshold, Global_vDisparityParams->thresholding_maxVal, Global_vDisparityParams->thresholding_type);

	new_rowRange.start = Global_uDisparityParams->thirdQuarterD;
	new_rowRange.end = Global_uDisparityParams->totalRows;
	Mat uDisparityImage_lastQuarterD = Global_uDisparityGlobals->uDisparityImage(new_rowRange, shifted_colRange);
	Mat uDisparityImage_lastQuarterD_thresholded;
	cv::adaptiveThreshold(uDisparityImage_lastQuarterD, uDisparityImage_lastQuarterD_thresholded, Global_uDisparityParams->thresholding_maxVal, Global_uDisparityParams->thresholding_adaptiveMethod, Global_uDisparityParams->thresholding_type, Global_uDisparityParams->thresholding_blockSize - 6, Global_uDisparityParams->thresholding_SubtractionConstant_C - 16);
	//cv::threshold(uDisparityImage_lastQuarterD, uDisparityImage_lastQuarterD_thresholded, Global_vDisparityParams->thresholding_threshold, Global_vDisparityParams->thresholding_maxVal, Global_vDisparityParams->thresholding_type);

	// hconcat(M1,M2,HM); // horizontal concatenation 	vconcat(M1, M2, VM); // vertical   concatenation
	uDisparityImage_zeroD.push_back(uDisparityImage_firstQuarterD); // I assume that this is the fastest way, check it later
	uDisparityImage_zeroD.push_back(uDisparityImage_secondQuarterD);
	uDisparityImage_zeroD.push_back(uDisparityImage_thirdQuarterD);
	uDisparityImage_zeroD.push_back(uDisparityImage_lastQuarterD);

	uDisparityImage_zeroD_thresholded.push_back(uDisparityImage_firstQuarterD_thresholded); // I assume that this is the fastest way, check it later
	uDisparityImage_zeroD_thresholded.push_back(uDisparityImage_secondQuarterD_thresholded);
	uDisparityImage_zeroD_thresholded.push_back(uDisparityImage_thirdQuarterD_thresholded);
	uDisparityImage_zeroD_thresholded.push_back(uDisparityImage_lastQuarterD_thresholded);

	uDisparityImage_zeroD_thresholded.copyTo(Global_uDisparityGlobals->uDisparityImage_thresholded);

	//zeroD = 7;
	//secondQuarterD = ((maxD + 1) / 2); = 32
	//firstQuarterD = ((maxD + 1) / 4); = 16
	//thirdQuarterD = (((maxD + 1) / 4) * 3); = 48



	//uDisparityImage = Global_uDisparityGlobals->uDisparityImage_shifted;

	//cv::adaptiveThreshold(Global_uDisparityGlobals->uDisparityImage_shifted, Global_uDisparityGlobals->uDisparityImage_thresholded, Global_uDisparityParams->thresholding_maxVal, Global_uDisparityParams->thresholding_adaptiveMethod, Global_uDisparityParams->thresholding_type, Global_uDisparityParams->thresholding_blockSize, Global_uDisparityParams->thresholding_SubtractionConstant_C);
	//uDisparityImage_thresholded1 = Global_uDisparityGlobals->uDisparityImage_thresholded;

	//cv::adaptiveThreshold(Global_uDisparityGlobals->uDisparityImage_shifted, uDisparityImage_thresholded2, Global_uDisparityParams->thresholding_maxVal, Global_uDisparityParams->thresholding_adaptiveMethod, Global_uDisparityParams->thresholding_type, Global_uDisparityParams->thresholding_blockSize-2, -25);

	copyMakeBorder(Global_uDisparityGlobals->uDisparityImage_thresholded, Global_uDisparityGlobals->uDisparityImage_thresholded, Global_uDisparityParams->shiftedStartRangeD, 0, 0, 0, BORDER_CONSTANT, 0); //resize to its original filling blanks with zero "0"
																																																			//}
																																																			//else
																																																			//{
																																																			//	cv::adaptiveThreshold(Global_uDisparityGlobals->uDisparityImage, Global_uDisparityGlobals->uDisparityImage_thresholded, Global_uDisparityParams->thresholding_maxVal, Global_uDisparityParams->thresholding_adaptiveMethod, Global_uDisparityParams->thresholding_type, Global_uDisparityParams->thresholding_blockSize, Global_uDisparityParams->thresholding_SubtractionConstant_C);
																																																			//}

	Global_uDisparityGlobals->uDisparityImage_labelled = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, CV_8UC1);
#pragma endregion Udisp Preparation

	Mat thresholded = Global_uDisparityGlobals->uDisparityImage_thresholded;

#pragma region Finding Lines By Labeling Alg
	//Check the zero quarter (where sensitivity of the camera is very low
	int isZeroQuarterFinished = false;
	isPreviousMaskOccupied = false;
	quarterNum = 0;
	u = 0;

	int nRemainCell_RowZeroquarter = Global_uDisparityParams->totalCols % Global_uDisparityParams->zeroD_MaskH;
	while (!isZeroQuarterFinished)
	{
		if (out_labelNum >= Global_maxNumOfObjectAllowed) { return; }

		if (u <= Global_uDisparityParams->totalCols)
		{
			isPreviousMaskOccupied = out_onceOccupied;
			subset_LabelingAlg(quarterNum, Global_uDisparityParams->zeroD_MaskH, Global_uDisparityParams->zeroD_MaskV, Global_uDisparityParams->mainFrame_zeroD_MaskV, u, d, isPreviousMaskOccupied, out_labelNum, out_onceOccupied, out_previousRightMostOccupiedCellIndex, Global_labeledObjectList);

			u += Global_uDisparityParams->zeroD_MaskH;
		}
		else // row has finished
		{
			if (nRemainCell_RowZeroquarter != 0)//if not! "nothing left at the end of the row"
			{
				subset_LabelingAlg(quarterNum, nRemainCell_RowZeroquarter, Global_uDisparityParams->zeroD_MaskV, Global_uDisparityParams->mainFrame_zeroD_MaskV, (u - Global_uDisparityParams->zeroD_MaskH), d, isPreviousMaskOccupied, out_labelNum, out_onceOccupied, out_previousRightMostOccupiedCellIndex, Global_labeledObjectList);
				out_onceOccupied = false;
				isPreviousMaskOccupied = false;
			}

			d += Global_uDisparityParams->zeroD_MaskV;

			if (d > Global_uDisparityParams->zeroD)
			{
				isZeroQuarterFinished = true;
			}

			u = 0;
		}
	}

	cv::Mat uDispDraw = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, CV_8UC3);
	uDispDraw = drawUDispObjects3(uDispDraw, Global_labeledObjectList, 0, 14);

	//Check the first quarter
	int isQuarterFinished = false;
	isPreviousMaskOccupied = false;
	quarterNum = 1;
	u = 0;

	int nRemainCell_Rowquarter = Global_uDisparityParams->totalCols % Global_uDisparityParams->quarterD_MaskH;
	while (!isQuarterFinished)
	{
		if (out_labelNum >= Global_maxNumOfObjectAllowed) { return; }

		if (u <= Global_uDisparityParams->totalCols)
		{
			isPreviousMaskOccupied = out_onceOccupied;
			subset_LabelingAlg(quarterNum, Global_uDisparityParams->quarterD_MaskH, Global_uDisparityParams->quarterD_MaskV, Global_uDisparityParams->mainFrame_quarterD_MaskV, u, d, isPreviousMaskOccupied, out_labelNum, out_onceOccupied, out_previousRightMostOccupiedCellIndex, Global_labeledObjectList);

			u += Global_uDisparityParams->quarterD_MaskH;
		}
		else // row has finished
		{
			if (nRemainCell_Rowquarter != 0)//if not! "nothing left at the end of the row"
			{
				subset_LabelingAlg(quarterNum, nRemainCell_Rowquarter, Global_uDisparityParams->quarterD_MaskV, Global_uDisparityParams->mainFrame_quarterD_MaskV, (u - Global_uDisparityParams->quarterD_MaskH), d, isPreviousMaskOccupied, out_labelNum, out_onceOccupied, out_previousRightMostOccupiedCellIndex, Global_labeledObjectList);
				out_onceOccupied = false;
				isPreviousMaskOccupied = false;
			}

			d += Global_uDisparityParams->quarterD_MaskV;

			if (d > Global_uDisparityParams->firstQuarterD)
			{
				isQuarterFinished = true;
			}

			u = 0;
		}
	}

	cv::Mat uDispDraw2 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, CV_8UC3);
	uDispDraw2 = drawUDispObjects3(uDispDraw2, Global_labeledObjectList, 14, 20);

	//Check the second quarter
	bool isSecondQuarterFinished = false;
	isPreviousMaskOccupied = false;
	u = 0;
	quarterNum = 2;

	int nRemainCell_RowSecondquarter = Global_uDisparityParams->totalCols % Global_uDisparityParams->halfD_MaskH;
	while (!isSecondQuarterFinished)
	{
		if (out_labelNum >= Global_maxNumOfObjectAllowed) { return; }

		if (u <= Global_uDisparityParams->totalCols)
		{
			isPreviousMaskOccupied = out_onceOccupied;
			out_onceOccupied = false;
			subset_LabelingAlg(quarterNum, Global_uDisparityParams->halfD_MaskH, Global_uDisparityParams->halfD_MaskV, Global_uDisparityParams->mainFrame_halfD_MaskV, u, d, isPreviousMaskOccupied, out_labelNum, out_onceOccupied, out_previousRightMostOccupiedCellIndex, Global_labeledObjectList);

			u += Global_uDisparityParams->halfD_MaskH;
		}
		else // row has finished
		{
			if (nRemainCell_RowSecondquarter != 0)
			{
				subset_LabelingAlg(quarterNum, nRemainCell_RowSecondquarter, Global_uDisparityParams->halfD_MaskV, Global_uDisparityParams->mainFrame_halfD_MaskV, (u - Global_uDisparityParams->halfD_MaskH), d, isPreviousMaskOccupied, out_labelNum, out_onceOccupied, out_previousRightMostOccupiedCellIndex, Global_labeledObjectList);
				out_onceOccupied = false;
				isPreviousMaskOccupied = false;
			}
			d += Global_uDisparityParams->halfD_MaskV;

			if (d > Global_uDisparityParams->secondQuarterD)
			{
				isSecondQuarterFinished = true;
			}

			u = 0;
		}
	}


	//Check the third quarter
	bool isThirdQuarterFinished = false;
	isPreviousMaskOccupied = false;
	u = 0;
	quarterNum = 3;

	int nRemainCell_RowThirdquarter = Global_uDisparityParams->totalCols % Global_uDisparityParams->threequarterD_MaskH;
	while (!isThirdQuarterFinished)
	{
		if (out_labelNum >= Global_maxNumOfObjectAllowed) { return; }

		if (u <= Global_uDisparityParams->totalCols)
		{
			isPreviousMaskOccupied = out_onceOccupied;
			out_onceOccupied = false;
			subset_LabelingAlg(quarterNum, Global_uDisparityParams->threequarterD_MaskH, Global_uDisparityParams->threequarterD_MaskV, Global_uDisparityParams->mainFrame_threequarterD_MaskV, u, d, isPreviousMaskOccupied, out_labelNum, out_onceOccupied, out_previousRightMostOccupiedCellIndex, Global_labeledObjectList);

			u += Global_uDisparityParams->threequarterD_MaskH;
		}
		else// row has finished
		{
			if (nRemainCell_RowThirdquarter != 0)
			{
				subset_LabelingAlg(quarterNum, nRemainCell_RowThirdquarter, Global_uDisparityParams->threequarterD_MaskV, Global_uDisparityParams->mainFrame_threequarterD_MaskV, (u - Global_uDisparityParams->threequarterD_MaskH), d, isPreviousMaskOccupied, out_labelNum, out_onceOccupied, out_previousRightMostOccupiedCellIndex, Global_labeledObjectList);
				out_onceOccupied = false;
				isPreviousMaskOccupied = false;
			}
			d += Global_uDisparityParams->threequarterD_MaskV;

			if (d > Global_uDisparityParams->thirdQuarterD)
			{
				isThirdQuarterFinished = true;
			}

			u = 0;
		}
	}

	//Check the last quarter
	bool isLastQuarterFinished = false;
	isPreviousMaskOccupied = false;
	u = 0;
	quarterNum = 4;

	int nRemainCell_RowLastquarter = Global_uDisparityParams->totalCols % Global_uDisparityParams->maxD_MaskH;
	int nRemainCell_ColumnLastquarter = (Global_uDisparityParams->totalRows - d) % Global_uDisparityParams->maxD_MaskV; //Check if we finish all rows
	while (!isLastQuarterFinished)
	{
		if (out_labelNum >= Global_maxNumOfObjectAllowed) { return; }

		if (u <= Global_uDisparityParams->totalCols)
		{
			isPreviousMaskOccupied = out_onceOccupied;
			out_onceOccupied = false;
			subset_LabelingAlg(quarterNum, Global_uDisparityParams->maxD_MaskH, Global_uDisparityParams->maxD_MaskV, Global_uDisparityParams->mainFrame_maxD_MaskV, u, d, isPreviousMaskOccupied, out_labelNum, out_onceOccupied, out_previousRightMostOccupiedCellIndex, Global_labeledObjectList);

			u += Global_uDisparityParams->maxD_MaskH;
		}
		else // row has finished
		{
			if (nRemainCell_RowLastquarter != 0)
			{
				subset_LabelingAlg(quarterNum, nRemainCell_RowLastquarter, Global_uDisparityParams->maxD_MaskV, Global_uDisparityParams->mainFrame_maxD_MaskV, (u - Global_uDisparityParams->maxD_MaskH), d, isPreviousMaskOccupied, out_labelNum, out_onceOccupied, out_previousRightMostOccupiedCellIndex, Global_labeledObjectList);
				out_onceOccupied = false;
				isPreviousMaskOccupied = false;
			}

			d += Global_uDisparityParams->maxD_MaskV;

			if (d > Global_uDisparityParams->maxD)
			{
				if (nRemainCell_ColumnLastquarter != 0)
				{
					u = 0;
					subset_LabelingAlg(quarterNum, Global_uDisparityParams->maxD_MaskH, nRemainCell_ColumnLastquarter, nRemainCell_ColumnLastquarter, u, (d - Global_uDisparityParams->maxD_MaskV), isPreviousMaskOccupied, out_labelNum, out_onceOccupied, out_previousRightMostOccupiedCellIndex, Global_labeledObjectList);
					out_onceOccupied = false;
					isPreviousMaskOccupied = false;
				}

				isLastQuarterFinished = true;
			}

			u = 0;
		}
	}

	//close again the last object in case no cell is provided after the last object detected
	Global_labeledObjectList[out_labelNum - 1].xHigh_Index = out_previousRightMostOccupiedCellIndex;
	Global_labeledObjectList[out_labelNum - 1].setUdispObject(*Global_uDisparityParams);
	//close again the last object in case no cell is provided after the last object detected

	Global_currentNumOfObjects = out_labelNum;

	cv::Mat uDispDraw4 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, CV_8UC3);
	uDispDraw4 = drawUDispObjects(uDispDraw4, Global_labeledObjectList);

	//cout << "hey" << endl;

#pragma endregion Finding Lines By Labeling Alg	

}

void inline StereoImageBuffer::subset_LabelingAlg(int quarterNum, int horizontalMask, int verticalMask, int mainFrame_verticalMask, int current_u, int current_d, bool isPreviousMaskOccupied, int &out_labelNum, bool& out_onceOccupied, int& out_previousRightMostOccupiedCellIndex, std::vector<MyObject, NAlloc<MyObject>>& myGlobal_currentObjectList/*std::vector<MyObject>& myGlobal_currentObjectList*/) const
{
	out_onceOccupied = false;

	const unsigned char* cell_uDisparity_thresholded;
	unsigned char* cell_uDisparity_labeled;

	cv::Range rowRange, colRange;
	colRange.start = current_u; colRange.end = current_u + horizontalMask;//horizontal limit of the mask
	rowRange.start = current_d; rowRange.end = current_d + verticalMask;//vertical limit of the mask , we only go down, not up

	if (colRange.end > Global_uDisparityParams->totalCols) { colRange.end = Global_uDisparityParams->totalCols; }
	else { if (colRange.start < 0) { colRange.start = 0; } }

	if (rowRange.end > Global_uDisparityParams->totalRows) { rowRange.end = Global_uDisparityParams->totalRows; }
	else { if (rowRange.start < 0) { rowRange.start = 0; } }

	bool isOccupied = false;
	bool isNewCell = false;

	int rightMostOccupiedCellIndex = 0;
	int leftMostOccupiedCellIndex = colRange.end;
	int upperPointOccupiedCellIndex = rowRange.end;
	int lowerPointOccupiedCellIndex = 0;

	Mat Labeled = Global_uDisparityGlobals->uDisparityImage_labelled;
	Mat thresholded = Global_uDisparityGlobals->uDisparityImage_thresholded;
	Mat thresholdedMask = Global_uDisparityGlobals->uDisparityImage_thresholded(rowRange, colRange);
	Mat labeledMask = Global_uDisparityGlobals->uDisparityImage_labelled(rowRange, colRange);

	for (int d = rowRange.start; d < rowRange.end; d++)
	{
		cell_uDisparity_thresholded = Global_uDisparityGlobals->uDisparityImage_thresholded.ptr<const unsigned char>(d);

		for (int u = colRange.start; u < colRange.end; u++)
		{
			isOccupied = cell_uDisparity_thresholded[u];

			if (isOccupied) //then label this cell
			{
				if (((!isPreviousMaskOccupied) && (!isNewCell)) /*|| (isPreviousMaskOccupied && (u> Global_imgMiddleCol && leftMostOccupiedCellIndex <= Global_imgMiddleCol) && d <= Global_uDisparityParams->maxDOfSideBuildings && (!isNewCell))*/)
				{
					out_labelNum = out_labelNum + 1; //we have started a new object, so save the leftmost(xLow) point
					isNewCell = true;

					if (out_labelNum > Global_maxNumOfObjectAllowed) { out_onceOccupied = false; break; }
				}

				if (!out_onceOccupied) out_onceOccupied = true;

				cell_uDisparity_labeled = Global_uDisparityGlobals->uDisparityImage_labelled.ptr<unsigned char>(d);
				cell_uDisparity_labeled[u] = out_labelNum;

				if (u > rightMostOccupiedCellIndex) rightMostOccupiedCellIndex = u; //save the rightmost point of the mask
				if (u < leftMostOccupiedCellIndex) leftMostOccupiedCellIndex = u; //save the leftmost point of the mask
				if (d < upperPointOccupiedCellIndex) upperPointOccupiedCellIndex = d; //save the upper point (lowDisp) of the mask
				if (d > lowerPointOccupiedCellIndex) lowerPointOccupiedCellIndex = d; //save the lower point (highDisp) of the mask
			}
		}
	}

	//Mat labeledMask2 = Global_uDisparityGlobals->uDisparityImage_labelled(rowRange, colRange);

	if (out_onceOccupied)
	{
		if (isNewCell) //if this is the first mask of the object
		{
#pragma region Get Started with the Current Object
			myGlobal_currentObjectList[out_labelNum - 1].xLow_Index = leftMostOccupiedCellIndex;
			myGlobal_currentObjectList[out_labelNum - 1].quarterNum = quarterNum;
			myGlobal_currentObjectList[out_labelNum - 1].maskH = horizontalMask;
			myGlobal_currentObjectList[out_labelNum - 1].maskV = verticalMask;
			myGlobal_currentObjectList[out_labelNum - 1].labelNum = out_labelNum - 1;
			myGlobal_currentObjectList[out_labelNum - 1].xLow_Index_lowDisp = upperPointOccupiedCellIndex;
			myGlobal_currentObjectList[out_labelNum - 1].xLow_Index_highDisp = lowerPointOccupiedCellIndex;
#pragma endregion Get Started with the Current Object

#pragma region Set the previous Object
			if ((out_labelNum - 1) != 0)
			{
				int previousObjectIndex = out_labelNum - 2;
				int len = out_previousRightMostOccupiedCellIndex - myGlobal_currentObjectList[previousObjectIndex].xLow_Index;
				//int slope = myGlobal_currentObjectList[previousObject].yhighDisp_Index - myGlobal_currentObjectList[previousObject].ylowDisp_Index;

				if (len < (myGlobal_currentObjectList[previousObjectIndex].maskH / 3)/*|| (slope == 0 && myGlobal_currentObjectList[previousObject].yhighDisp_Index==0)*/) //filter object
				{
					myGlobal_currentObjectList[previousObjectIndex].xHigh_Index = out_previousRightMostOccupiedCellIndex;
					myGlobal_currentObjectList[previousObjectIndex].isDeleted = true;
				}
				else
				{
					myGlobal_currentObjectList[previousObjectIndex].xHigh_Index = out_previousRightMostOccupiedCellIndex;
					myGlobal_currentObjectList[previousObjectIndex].setUdispObject(*Global_uDisparityParams);
				}
			}
#pragma endregion Set the previous Object
		}

#pragma region Continue with the Current Object
		out_previousRightMostOccupiedCellIndex = rightMostOccupiedCellIndex; //had to be after "if (isNewCell)"

																			 //check every occupied mask to set lower and upper points
		if (upperPointOccupiedCellIndex < myGlobal_currentObjectList[out_labelNum - 1].ylowDisp_Index)
		{
			myGlobal_currentObjectList[out_labelNum - 1].ylowDisp_Index = upperPointOccupiedCellIndex; //highest point (low disp) of the mask
		}

		if (lowerPointOccupiedCellIndex > myGlobal_currentObjectList[out_labelNum - 1].yhighDisp_Index) //check evey mask to set lower and upper points
		{
			myGlobal_currentObjectList[out_labelNum - 1].yhighDisp_Index = lowerPointOccupiedCellIndex; //lowest point (high disp) of the mask
		}
		//check every occupied mask to set lower and upper points		
#pragma endregion Continue with the Current Object
	}//if the mask is once occupied

}

void inline StereoImageBuffer::merge_MaskVObjects_From_UDisparity_By_Labeling_Alg(std::vector<MyObject, NAlloc<MyObject>>& previousObjectList)
{
	int extended_obj_xLow, extended_obj_xHigh, total_height, labelNum = 0;
	bool use_search_object_params;
	bool use_search_object_xlowD;

	Global_mergedObjectList1.clear();
	Global_mergedObjectList1.resize(Global_currentNumOfObjects);//allocate the storage
	MyObject obj, search_obj;

	//cv::Mat colored_Disparity;
	//cv::cvtColor(Global_Current_DenseDisparity_Image, colored_Disparity, CV_GRAY2RGB);

	//cv::Mat originalIMG = Global_Current_Left_Image;

	//cv::Mat vDisparity = Global_vDisparityGlobals->vDisparityImage;
	//cv::Mat uDisparity = Global_uDisparityGlobals->uDisparityImage;
	//cv::Mat uDisparity_thresholded = Global_uDisparityGlobals->uDisparityImage_thresholded;

	//cv::Mat colorV;
	//cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV, CV_GRAY2BGR);
	//cv::line(colorV, Point(0, Global_vDisparityGlobals->bestRoadLine_vIntersection), Point(Global_vDisparityGlobals->bestRoadLine_dInterSection, Global_vDisparityParams->totalRows - 1), cv::Scalar(0, 0, 255));
	//cv::line(colorV, Point(Global_vDisparityGlobals->piecewiseBestLine[0], Global_vDisparityGlobals->piecewiseBestLine[1]), Point(Global_vDisparityGlobals->piecewiseBestLine[2], Global_vDisparityGlobals->piecewiseBestLine[3]), cv::Scalar(0, 255, 0));

	//cv::Mat uDispDraw = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, colorV.type());
	//uDispDraw = drawUDispObjects(uDispDraw, previousObjectList);

	for (int obj_indx = 0; obj_indx < Global_currentNumOfObjects; obj_indx++)
	{
		obj = previousObjectList[obj_indx];

		if (!obj.isDeleted)
		{
			//printObject(obj);
			for (int search_obj_indx = 0; search_obj_indx < Global_currentNumOfObjects; search_obj_indx++)
			{
				if (search_obj_indx != obj_indx)
				{
					search_obj = previousObjectList[search_obj_indx];

					if (!search_obj.isDeleted)
					{
						//get new total height
						int lowerPoint, upperPoint, leftPoint, rightPoint;
						use_search_object_params = false;
						use_search_object_xlowD = false;

						if (search_obj.yhighDisp_Index > obj.yhighDisp_Index) { lowerPoint = search_obj.yhighDisp_Index;	use_search_object_params = true; }
						else { lowerPoint = obj.yhighDisp_Index; }

						if (search_obj.ylowDisp_Index < obj.ylowDisp_Index) { upperPoint = search_obj.ylowDisp_Index; use_search_object_xlowD = true; }
						else { upperPoint = obj.ylowDisp_Index; }

						total_height = lowerPoint - upperPoint + 1;
						int maxV;

						if (obj.isMainFrame || search_obj.isMainFrame)
						{
							if (search_obj.isMainFrame)
								maxV = search_obj.mainFrame_maskV;
							else
								maxV = obj.mainFrame_maskV;
						}
						else
						{
							maxV = search_obj.maskV;
						}

						if (total_height <= maxV) //if merging process does not exceed the maximum height restriction (!!! maskV changed obj.maskV to search_obj.maskv!!!)
						{
							//check x coordinates if two object intersects
							extended_obj_xLow = (obj.xLow_Index - obj.maskH + 1);
							extended_obj_xHigh = (obj.xHigh_Index + obj.maskH - 1);

							if ((search_obj.xLow_Index <= extended_obj_xLow && search_obj.xHigh_Index >= extended_obj_xLow) || (search_obj.xLow_Index >= extended_obj_xLow && search_obj.xLow_Index <= extended_obj_xHigh))
							{
								if (search_obj.xHigh_Index > obj.xHigh_Index) { rightPoint = search_obj.xHigh_Index; }
								else { rightPoint = obj.xHigh_Index; }

								if (search_obj.xLow_Index < obj.xLow_Index) { leftPoint = search_obj.xLow_Index; }
								else { leftPoint = obj.xLow_Index; }

								if (!((leftPoint < Global_uDisparityParams->middleCol && rightPoint >= Global_uDisparityParams->middleCol) && upperPoint <= Global_uDisparityParams->maxDOfSideBuildings))//Try to avoid merging wall candidates located at opposite sites
								{
									updateMergingObjects(leftPoint, rightPoint, upperPoint, lowerPoint, use_search_object_params, use_search_object_xlowD, labelNum, previousObjectList[obj_indx], previousObjectList[search_obj_indx], Global_mergedObjectList1, true);

									//cout << obj_indx << " & " << search_obj_indx << " is merging" << endl;
									//Update local variables
									obj = previousObjectList[obj_indx]; // previousObjectList[obj_indx] is already updated in updateMergingObjects by using its reference &address
									search_obj = previousObjectList[search_obj_indx];// previousObjectList[search_obj_indx] is already updated in updateMergingObjects by using its reference &address
																					 //Update local variables

								}//if! merging wall candidates
							}//if x coordinates intersect
						}//if total_height <= obj.maskV
						 //check First if the object did not merge due to size of mask sliding windows
						 //else//inclined objects here
						 //{								

						 //}//else //inclined objects here
					}//if !search_obj.deleted
				} //if search_obj_indx != obj_indx					
			}//search for loop (sobj)

			if (!obj.isMerged)
			{
				//cout << obj_indx << " is not merged " << " NLab:" << labelNum << endl;
				previousObjectList[obj_indx].mergedLabelNum = labelNum;
				Global_mergedObjectList1[labelNum] = previousObjectList[obj_indx];
			}
			labelNum++;
			//Global_currentNumOfObjects = labelNum;
		}//if not obj is deleted
		else
		{
			// open for check 2cout << obj_indx << " is deleted!!! " << " NLab:" << labelNum << endl;
		}
	}//Main for loop (obj)

	Global_currentNumOfObjects = labelNum;

	//cv::Mat uDispDraw2 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, colorV.type());
	//uDispDraw2 = drawUDispObjects(uDispDraw2, Global_mergedObjectList1);
	//cout << "hey" << endl;
}

void inline StereoImageBuffer::merge_InclinedObjects_From_UDisparity_By_Labeling_Alg(std::vector<MyObject, NAlloc<MyObject>>& previousObjectList)
{
	//Now check the objects whose slope is greater than the VMask	
	bool upRight = false, downLeft, upLeft = false, downRight, xCoor_Check;
	int labelNum = 0;

	MyObject obj, search_obj;

	Global_mergedObjectList2.clear();
	Global_mergedObjectList2.resize(Global_currentNumOfObjects);

	cv::Mat originalLeftIMG = Global_Current_Left_Image;
	cv::Mat originalRightIMG = Global_Current_Right_Image;

	cv::Mat dDisparity = Global_Current_DenseDisparity_Image;
	cv::Mat vDisparity = Global_vDisparityGlobals->vDisparityImage;
	cv::Mat uDisparity = Global_uDisparityGlobals->uDisparityImage;
	cv::Mat uDisparity_thresholded = Global_uDisparityGlobals->uDisparityImage_thresholded;

	cv::Mat colorV;
	cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV, CV_GRAY2BGR);
	cv::line(colorV, Point(0, Global_vDisparityGlobals->bestRoadLine_vIntersection), Point(Global_vDisparityGlobals->bestRoadLine_dInterSection, Global_vDisparityParams->totalRows - 1), cv::Scalar(0, 0, 255));
	cv::line(colorV, Point(Global_vDisparityGlobals->piecewiseBestLine[0], Global_vDisparityGlobals->piecewiseBestLine[1]), Point(Global_vDisparityGlobals->piecewiseBestLine[2], Global_vDisparityGlobals->piecewiseBestLine[3]), cv::Scalar(0, 255, 0));

	//cv::Mat uDispDraw = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, colorV.type());
	//uDispDraw = drawUDispObjects(uDispDraw, previousObjectList);

	for (int obj_indx = 0; obj_indx < Global_currentNumOfObjects; obj_indx++)
	{
		obj = previousObjectList[obj_indx];

		if (!obj.isDeleted)
		{
			//printObject(obj);
			for (int search_obj_indx = 0; search_obj_indx < Global_currentNumOfObjects; search_obj_indx++)
			{
				if (search_obj_indx != obj_indx)// we only have to check down-left or down-right search_obj's which, their labelNum is bigger
				{
					search_obj = previousObjectList[search_obj_indx];

					if (!search_obj.isDeleted)
					{
						if (!(obj.isLeftLeaning ^ search_obj.isLeftLeaning) || (obj.isStraightLine || search_obj.isStraightLine)) //XOR gate. if they are not at opposite angles
						{
							if ((obj.isLeftLeaning && search_obj.isLeftLeaning) || (obj.isLeftLeaning && search_obj.isStraightLine) || (obj.isStraightLine && search_obj.isLeftLeaning)) //if both objs are on the left side
							{
								downLeft = ((search_obj.ylowDisp_Index >= obj.yhighDisp_Index) && (search_obj.ylowDisp_Index <= (obj.yhighDisp_Index + round((obj.maskV + 0.1) / 2))));
								//downLeft = ((search_obj.ylowDisp_Index >= obj.yhighDisp_Index) && (abs(obj.yhighDisp_Index - search_obj.ylowDisp_Index) <= (obj.maskV / 2)));

								//abs(obj.yhighDisp_Index - search_obj.ylowDisp_Index) <= (obj.maskV / 2);

								if (!downLeft)
									upRight = ((search_obj.yhighDisp_Index <= obj.ylowDisp_Index) && (search_obj.yhighDisp_Index >= (obj.ylowDisp_Index - round((obj.maskV + 0.1) / 2))));


								if (upRight || downLeft) //if second obj's lowDisp in the range of the obj's(+maskV) highDisp
								{
									int obj_len = obj.getUdisp_Length();

									if (downLeft)
										xCoor_Check = ((search_obj.xLow_Index < obj.xLow_Index) && (search_obj.xHigh_Index <= (obj.xLow_Index + (obj_len / 2))) && (search_obj.xHigh_Index >= (obj.xLow_Index - obj.maskH + 1)));
									else
										xCoor_Check = ((search_obj.xHigh_Index > obj.xHigh_Index) && (search_obj.xLow_Index >= (obj.xLow_Index + (obj_len / 2))) && (search_obj.xLow_Index <= (obj.xHigh_Index + obj.maskH - 1)));

									if (xCoor_Check)
									{
										if (abs(obj.udisp_degree - search_obj.udisp_degree) < search_obj.inclinedObjAnglePrecision) //changed to search_obj AnglePrecision
										{
											if (downLeft)
											{
												updateMergingObjects(search_obj.xLow_Index, obj.xHigh_Index, obj.ylowDisp_Index, search_obj.yhighDisp_Index, true, true, labelNum, previousObjectList[obj_indx], previousObjectList[search_obj_indx], Global_mergedObjectList2, false);
												//cout << "M2DL: " << obj_indx << "(" << obj.xLow_Index << "," << obj.xHigh_Index << "," << obj.ylowDisp_Index << "," << obj.yhighDisp_Index << "," << obj.udisp_degree << ") and " << search_obj_indx << "(" << search_obj.xLow_Index << "," << search_obj.xHigh_Index << "," << search_obj.ylowDisp_Index << "," << search_obj.yhighDisp_Index << "," << search_obj.udisp_degree << ")" << " -> " << " X:" << previousObjectList[obj_indx].xLow_Index << "," << previousObjectList[obj_indx].xHigh_Index << " Y:" << previousObjectList[obj_indx].ylowDisp_Index << "," << previousObjectList[obj_indx].yhighDisp_Index << " NLab:" << previousObjectList[obj_indx].mergedLabelNumInclined << endl << endl;
											}
											else
											{
												updateMergingObjects(obj.xLow_Index, search_obj.xHigh_Index, search_obj.ylowDisp_Index, obj.yhighDisp_Index, false, true, labelNum, previousObjectList[obj_indx], previousObjectList[search_obj_indx], Global_mergedObjectList2, false);
												//cout << "M2UR: " << obj_indx << "(" << obj.xLow_Index << "," << obj.xHigh_Index << "," << obj.ylowDisp_Index << "," << obj.yhighDisp_Index << "," << obj.udisp_degree << ") and " << search_obj_indx << "(" << search_obj.xLow_Index << "," << search_obj.xHigh_Index << "," << search_obj.ylowDisp_Index << "," << search_obj.yhighDisp_Index << "," << search_obj.udisp_degree << ")" << " -> " << " X:" << previousObjectList[obj_indx].xLow_Index << "," << previousObjectList[obj_indx].xHigh_Index << " Y:" << previousObjectList[obj_indx].ylowDisp_Index << "," << previousObjectList[obj_indx].yhighDisp_Index << " NLab:" << previousObjectList[obj_indx].mergedLabelNumInclined << endl << endl;
											}

											//Update local variables
											obj = previousObjectList[obj_indx]; // myObjectList[obj_indx] is already updated in updateMergingObjects by using its reference address
											search_obj = previousObjectList[search_obj_indx];// myObjectList[search_obj_indx] is already updated in updateMergingObjects by using its reference address
																							 //Update local variables
										}
										else
										{
											// open for check 2cout << fixed << std::setprecision(2) << "M2L!: " << obj_indx << "(" << obj.xLow_Index << "," << obj.xHigh_Index << "," << obj.ylowDisp_Index << "," << obj.yhighDisp_Index << ") and " << search_obj_indx << "(" << search_obj.xLow_Index << "," << search_obj.xHigh_Index << "," << search_obj.ylowDisp_Index << "," << search_obj.yhighDisp_Index << ")" << " is not merged " << " obj:" << obj.udisp_degree << " sobj:" << search_obj.udisp_degree << endl;
										}
									}// (endif-left) X coordinates check
								}// (endif-left) Y coordinates check
							}//if both objs are on the left side
							else
							{
								if ((!obj.isLeftLeaning) && (!search_obj.isLeftLeaning) || (!obj.isLeftLeaning && search_obj.isStraightLine) || (obj.isStraightLine && !search_obj.isLeftLeaning)) //if both objs are at the right sided angle
								{
									downRight = ((search_obj.ylowDisp_Index >= obj.yhighDisp_Index) && (search_obj.ylowDisp_Index <= (obj.yhighDisp_Index + round((obj.maskV + 0.1) / 2))));

									if (!downRight)
										upLeft = ((search_obj.yhighDisp_Index <= obj.ylowDisp_Index) && (search_obj.yhighDisp_Index >= (obj.ylowDisp_Index - round((obj.maskV + 0.1) / 2))));

									if (downRight || upLeft) //if second obj's lowDisp in the range of the obj's(+maskV) highDisp
									{
										int obj_len = obj.getUdisp_Length();

										if (downRight)
											xCoor_Check = ((search_obj.xHigh_Index > obj.xHigh_Index) && (search_obj.xLow_Index >= (obj.xLow_Index + (obj_len / 2))) && (search_obj.xLow_Index <= (obj.xHigh_Index + obj.maskH - 1)));
										else
											xCoor_Check = ((search_obj.xLow_Index < obj.xLow_Index) && (search_obj.xHigh_Index <= (obj.xLow_Index + (obj_len / 2))) && (search_obj.xHigh_Index >= (obj.xLow_Index - obj.maskH + 1)));

										//if ((search_obj_x2 > obj_x2) && (search_obj_x1 >= (obj_x1 + (obj_len / 4))) && (search_obj_x1 <= (obj_x2 + obj.maskH - 1)))
										if (xCoor_Check)
										{
											if (abs(obj.udisp_degree - search_obj.udisp_degree) < search_obj.inclinedObjAnglePrecision)
											{
												if (downRight)
												{
													updateMergingObjects(obj.xLow_Index, search_obj.xHigh_Index, obj.ylowDisp_Index, search_obj.yhighDisp_Index, true, false, labelNum, previousObjectList[obj_indx], previousObjectList[search_obj_indx], Global_mergedObjectList2, false);
													//cout << "M2DR: " << obj_indx << "(" << obj.xLow_Index << "," << obj.xHigh_Index << "," << obj.ylowDisp_Index << "," << obj.yhighDisp_Index << "," << obj.udisp_degree << ") and " << search_obj_indx << "(" << search_obj.xLow_Index << "," << search_obj.xHigh_Index << "," << search_obj.ylowDisp_Index << "," << search_obj.yhighDisp_Index << "," << search_obj.udisp_degree << ")" << " -> " << " X:" << previousObjectList[obj_indx].xLow_Index << "," << previousObjectList[obj_indx].xHigh_Index << " Y:" << previousObjectList[obj_indx].ylowDisp_Index << "," << previousObjectList[obj_indx].yhighDisp_Index << " NLab:" << previousObjectList[obj_indx].mergedLabelNumInclined << " left:" << previousObjectList[obj_indx].isLeftLeaning << " r:" << previousObjectList[obj_indx].isReverseSide << "xl:" << previousObjectList[obj_indx].xLow_Index_lowDisp << "xh:" << previousObjectList[obj_indx].xLow_Index_highDisp << endl << endl;
												}
												else
												{
													updateMergingObjects(search_obj.xLow_Index, obj.xHigh_Index, search_obj.ylowDisp_Index, obj.yhighDisp_Index, true, true, labelNum, previousObjectList[obj_indx], previousObjectList[search_obj_indx], Global_mergedObjectList2, false);
													//cout << "M2UL: " << obj_indx << "(" << obj.xLow_Index << "," << obj.xHigh_Index << "," << obj.ylowDisp_Index << "," << obj.yhighDisp_Index << "," << obj.udisp_degree << ") and " << search_obj_indx << "(" << search_obj.xLow_Index << "," << search_obj.xHigh_Index << "," << search_obj.ylowDisp_Index << "," << search_obj.yhighDisp_Index << "," << search_obj.udisp_degree << ")" << " -> " << " X:" << previousObjectList[obj_indx].xLow_Index << "," << previousObjectList[obj_indx].xHigh_Index << " Y:" << previousObjectList[obj_indx].ylowDisp_Index << "," << previousObjectList[obj_indx].yhighDisp_Index << " NLab:" << previousObjectList[obj_indx].mergedLabelNumInclined << endl << endl;
												}

												//Update local variables
												obj = previousObjectList[obj_indx]; // previousObjectList[obj_indx] is already updated in updateMergingObjects by using its reference address
												search_obj = previousObjectList[search_obj_indx];// previousObjectList[search_obj_indx] is already updated in updateMergingObjects by using its reference address
																								 //Update local variables
											}
											else
											{
												// open for check 2 cout << fixed << std::setprecision(2) << "M2R!: " << obj_indx << "(" << obj.xLow_Index << "," << obj.xHigh_Index << "," << obj.ylowDisp_Index << "," << obj.yhighDisp_Index << ") and " << search_obj_indx << "(" << search_obj.xLow_Index << "," << search_obj.xHigh_Index << "," << search_obj.ylowDisp_Index << "," << search_obj.yhighDisp_Index << ")" << " is not merged " << " obj:" << obj.udisp_degree << " sobj:" << search_obj.udisp_degree << endl;
											}
										}// (endif-left) X coordinates check
									}// (endif-left) Y coordinates check
								}//if both objs are on the right side
							}//else //if both objs are NOT on the left side
						}//XOR gate. if they are not at different sides
					}//if !search_obj.deleted
				}//if search_obj_indx != obj_indx	
			}//search for loop (sobj)

			if (!obj.isMergedInclined)
			{
				//cout << obj_indx << " is not merged " << " NLab:" << labelNum << endl;
				previousObjectList[obj_indx].mergedLabelNumInclined = labelNum;
				Global_mergedObjectList2[labelNum] = previousObjectList[obj_indx];
			}
			labelNum++;
		}//if not obj is deleted
		else
		{
			// open for check 2 cout << obj_indx << " is deleted!!! " << " NLab:" << labelNum << endl;
		}
	}//Main for loop (obj)
	 //Now check the objects whose slope is greater than the VMask

	Global_currentNumOfObjects = labelNum;
	//cv::Mat uDispDraw2 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, colorV.type());
	//uDispDraw2 = drawUDispObjects(uDispDraw2, Global_mergedObjectList2);
	//cout << "hey" << endl;
}

void inline StereoImageBuffer::updateMergingObjects(int new_xLow, int new_xHigh, int new_ylow, int new_yhigh, bool use_search_object_params, bool use_search_object_xlowD, int& out_currentLabelNum, MyObject &out_obj, MyObject &out_sobj, std::vector<MyObject, NAlloc<MyObject>>& out_newObjectList, bool straightLineMerge) const
{
	out_obj.xLow_Index = new_xLow;
	out_obj.xHigh_Index = new_xHigh;
	out_obj.ylowDisp_Index = new_ylow;
	out_obj.yhighDisp_Index = new_yhigh;

	if (use_search_object_params)
	{
		out_obj.quarterNum = out_sobj.quarterNum;
	}

	if (use_search_object_xlowD)//if sobj xlow is smaller than the obj's xlow save the most left point
	{
		out_obj.xLow_Index_lowDisp = out_sobj.xLow_Index_lowDisp;
		out_obj.xLow_Index_highDisp = out_sobj.xLow_Index_highDisp;
	}

	out_obj.setUdispObject(*Global_uDisparityParams);

	if (straightLineMerge)//merging lines whose height <= maskV
	{
		if (out_sobj.labelNum < out_obj.labelNum)//if sobj_index < obj_index and sobj is not merged or deleted due to some reasons
		{
			out_currentLabelNum--;

			out_sobj.isDeleted = true;
			out_sobj.isMerged = true;

			out_obj.mergedLabelNum = out_sobj.mergedLabelNum;
			out_obj.isMerged = true;

			out_newObjectList[out_sobj.mergedLabelNum] = out_obj;
		}
		else
		{
			if (out_obj.isMerged) //if merged before
			{
				out_sobj.isDeleted = true;
				out_sobj.isMerged = true;
				out_sobj.mergedLabelNum = out_obj.mergedLabelNum;

				out_newObjectList[out_obj.mergedLabelNum] = out_obj;
			}
			else
			{
				out_sobj.isDeleted = true;
				out_sobj.isMerged = true;
				out_sobj.mergedLabelNum = out_currentLabelNum;

				out_obj.isMerged = true;
				out_obj.mergedLabelNum = out_currentLabelNum;

				out_newObjectList[out_currentLabelNum] = out_obj;
			}
		}
	}
	else//if inclined merge, //merging lines whose height > maskV already defined by labeling alg.
	{
		if (out_sobj.labelNum < out_obj.labelNum)//if sobj_index < obj_index and sobj is not merged or deleted due to some reasons
		{
			//cout << "sobj.labelNum < obj.labelNum " << sobj.labelNum << " - " << obj.labelNum << " Img:" << getCurrentImageNum().str() << endl;
			out_currentLabelNum--;

			out_sobj.isDeleted = true;
			out_sobj.isMergedInclined = true;

			out_obj.mergedLabelNumInclined = out_sobj.mergedLabelNumInclined;
			out_obj.isMergedInclined = true;

			out_newObjectList[out_sobj.mergedLabelNumInclined] = out_obj;
		}
		else
		{
			if (out_obj.isMergedInclined) //if merged before
			{
				out_sobj.isDeleted = true;
				out_sobj.isMergedInclined = true;
				out_sobj.mergedLabelNumInclined = out_obj.mergedLabelNumInclined;

				out_newObjectList[out_obj.mergedLabelNumInclined] = out_obj;
			}
			else
			{
				out_sobj.isDeleted = true;
				out_sobj.isMergedInclined = true;
				out_sobj.mergedLabelNumInclined = out_currentLabelNum;

				out_obj.isMergedInclined = true;
				out_obj.mergedLabelNumInclined = out_currentLabelNum;

				out_newObjectList[out_currentLabelNum] = out_obj;
			}
		}
	}
}

void inline StereoImageBuffer::set_SideBuildingsIfAny_And_Filter(std::vector<MyObject, NAlloc<MyObject>>& previousObjectList)
{
	//Try to find the left and right sides of the road if there are buildings
	float midX_old_left, midX_new_left, midX_old_right, midX_new_right;
	MyObject obj;

	//cv::Mat originalLeftIMG = Global_Current_Left_Image;
	//cv::Mat originalRightIMG = Global_Current_Right_Image;

	//cv::Mat dDisparity = Global_Current_DenseDisparity_Image;
	//cv::Mat vDisparity = Global_vDisparityGlobals->vDisparityImage;
	//cv::Mat uDisparity = Global_uDisparityGlobals->uDisparityImage;
	//cv::Mat uDisparity_thresholded = Global_uDisparityGlobals->uDisparityImage_thresholded;

	//cv::Mat colorV;
	//cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV, CV_GRAY2BGR);
	//cv::line(colorV, Point(0, Global_vDisparityGlobals->bestRoadLine_vIntersection), Point(Global_vDisparityGlobals->bestRoadLine_dInterSection, Global_vDisparityParams->totalRows - 1), cv::Scalar(0, 0, 255));
	//cv::line(colorV, Point(Global_vDisparityGlobals->piecewiseBestLine[0], Global_vDisparityGlobals->piecewiseBestLine[1]), Point(Global_vDisparityGlobals->piecewiseBestLine[2], Global_vDisparityGlobals->piecewiseBestLine[3]), cv::Scalar(0, 255, 0));

	//cv::Mat uDispDraw = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, colorV.type());
	//uDispDraw = drawUDispObjects(uDispDraw, previousObjectList);

	for (int obji = 0; obji < Global_currentNumOfObjects; obji++)
	{
		obj = previousObjectList[obji];

		if (obj.ylowDisp_Index <= Global_uDisparityParams->maxDOfSideBuildings && obj.yhighDisp_Index <= Global_uDisparityParams->secondQuarterD && obj.getUdisp_Length() > Global_uDisparityParams->minLengthOfSideBuildings) // ??? && abs(obj.udisp_degree) <= abs(Global_vDisparityGlobals->bestRoadLine_m) ???
		{
			if (obj.isLeftLeaning)
			{
				if (Global_uDisparityGlobals->sideLineLeftIndex == -1)
				{
					Global_uDisparityGlobals->sideLineLeftIndex = obji;
				}
				else
				{
					if (obj.ylowDisp_Index <= previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].ylowDisp_Index)
					{
						Global_uDisparityGlobals->sideLineLeftIndex = obji;
					}
				}
			}
			else
			{
				if (Global_uDisparityGlobals->sideLineRightIndex == -1)
				{
					Global_uDisparityGlobals->sideLineRightIndex = obji;
				}
				else
				{
					if (obj.ylowDisp_Index <= previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].ylowDisp_Index)
					{
						Global_uDisparityGlobals->sideLineRightIndex = obji;
					}
				}
			}
		}
	}

	//cout << endl;
	//cout << "left:" << Global_uDisparityGlobals->sideLineLeftIndex << " right:" << Global_uDisparityGlobals->sideLineRightIndex;

	//try to find left and right buildings
	if (Global_uDisparityGlobals->sideLineLeftIndex != -1 && Global_uDisparityGlobals->sideLineRightIndex != -1)//İf we found both then get the intersection of them
	{
		//set leftleaning line lower disp
		previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].isSideBuilding = true;
		Global_uDisparityGlobals->sideLineLeft_m = previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].udisp_slope;
		Global_uDisparityGlobals->sideLineLeft_b = previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].udisp_slope_b;

		if (previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].ylowDisp_Index > 4)
		{
			previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].ylowDisp_Index = 4; //since 0-1 disparities are so fuzzy
			previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].xHigh_Index = (previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].ylowDisp_Index - Global_uDisparityGlobals->sideLineLeft_b) / Global_uDisparityGlobals->sideLineLeft_m;
		}


		if (previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].xHigh_Index >= Global_uDisparityParams->totalCols)
		{
			previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].xHigh_Index = Global_uDisparityParams->totalCols - 1;
			previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].ylowDisp_Index = round(Global_uDisparityGlobals->sideLineLeft_m*previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].xHigh_Index + Global_uDisparityGlobals->sideLineLeft_b);
		}

		previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].setUdispObject(*Global_uDisparityParams);

		Global_uDisparityGlobals->infinityLeftPointX = previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].xHigh_Index;
		Global_uDisparityGlobals->infinityLeftPointY = previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].ylowDisp_Index;

		Global_dDisparityGlobals->infinityUlineLeftPointX = Global_uDisparityGlobals->infinityLeftPointX;
		Global_dDisparityGlobals->infinityUlineLeftPointY = get_v_From_VDisparity_RoadProfile(Global_uDisparityGlobals->infinityLeftPointY); //round(Global_uDisparityGlobals->infinityLeftPointY*Global_vDisparityGlobals->bestRoadLine_m + Global_vDisparityGlobals->bestRoadLine_b);

																																			 //set rightleaning line lower disp
		previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].isSideBuilding = true;
		Global_uDisparityGlobals->sideLineRight_m = previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].udisp_slope;
		Global_uDisparityGlobals->sideLineRight_b = previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].udisp_slope_b;

		if (previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].ylowDisp_Index > 4)
		{
			previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].ylowDisp_Index = 4;
			previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].xLow_Index = (previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].ylowDisp_Index - Global_uDisparityGlobals->sideLineRight_b) / Global_uDisparityGlobals->sideLineRight_m;
		}


		if (previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].xLow_Index < 0)
		{
			previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].xLow_Index = 0;
			previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].ylowDisp_Index = round(Global_uDisparityGlobals->sideLineRight_m*previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].xLow_Index + Global_uDisparityGlobals->sideLineRight_b);
		}

		previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].xLow_Index_lowDisp = previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].ylowDisp_Index;//avoid udispsetup to reverse it

		Global_uDisparityGlobals->infinityRightPointX = previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].xLow_Index;
		Global_uDisparityGlobals->infinityRightPointY = previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].ylowDisp_Index;

		Global_dDisparityGlobals->infinityUlineRightPointX = Global_uDisparityGlobals->infinityRightPointX;
		Global_dDisparityGlobals->infinityUlineRightPointY = get_v_From_VDisparity_RoadProfile(Global_uDisparityGlobals->infinityRightPointY); //round(Global_uDisparityGlobals->infinityRightPointY*Global_vDisparityGlobals->bestRoadLine_m + Global_vDisparityGlobals->bestRoadLine_b);


		if (!(previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].xLow_Index < previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].xHigh_Index))
		{
			Global_uDisparityGlobals->infinityCommonPointX = (previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].xHigh_Index + previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].xLow_Index) / 2;
			Global_uDisparityGlobals->infinityCommonPointY = 0;
			Global_dDisparityGlobals->infinityUlineCommonPointX = Global_uDisparityGlobals->infinityCommonPointX;
			Global_dDisparityGlobals->infinityUlineCommonPointY = get_v_From_VDisparity_RoadProfile(Global_uDisparityGlobals->infinityCommonPointY); //round(Global_uDisparityGlobals->infinityCommonPointY*Global_vDisparityGlobals->bestRoadLine_m + Global_vDisparityGlobals->bestRoadLine_b);

		}
		else
		{
			//get the intersection of them
			int intersection_X = (Global_uDisparityGlobals->sideLineRight_b - Global_uDisparityGlobals->sideLineLeft_b) / (Global_uDisparityGlobals->sideLineLeft_m - Global_uDisparityGlobals->sideLineRight_m);
			int intersection_Y = round(Global_uDisparityGlobals->sideLineLeft_m*intersection_X + Global_uDisparityGlobals->sideLineLeft_b);

			if (intersection_Y >= 0 && intersection_Y < Global_uDisparityParams->totalRows)
			{
				previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].xHigh_Index = intersection_X;
				previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].ylowDisp_Index = intersection_Y;

				previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].xLow_Index = intersection_X;
				previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].ylowDisp_Index = intersection_Y;

				Global_uDisparityGlobals->infinityCommonPointX = intersection_X;
				Global_uDisparityGlobals->infinityCommonPointY = intersection_Y;
				Global_dDisparityGlobals->infinityUlineCommonPointX = Global_uDisparityGlobals->infinityCommonPointX;
				Global_dDisparityGlobals->infinityUlineCommonPointY = get_v_From_VDisparity_RoadProfile(Global_uDisparityGlobals->infinityCommonPointY); //round(Global_uDisparityGlobals->infinityCommonPointY*Global_vDisparityGlobals->bestRoadLine_m + Global_vDisparityGlobals->bestRoadLine_b);

			}
		}

		//previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].xLow_Index_highDisp =
		previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].setUdispObject(*Global_uDisparityParams);
	}
	else
	{
		if (Global_uDisparityGlobals->sideLineLeftIndex != -1)
		{
			//set leftleaning line lower disp
			previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].isSideBuilding = true;
			Global_uDisparityGlobals->sideLineLeft_m = previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].udisp_slope;
			Global_uDisparityGlobals->sideLineLeft_b = previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].udisp_slope_b;

			if (previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].ylowDisp_Index > 4)
			{
				previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].ylowDisp_Index = 4;
				previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].xHigh_Index = (previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].ylowDisp_Index - Global_uDisparityGlobals->sideLineLeft_b) / Global_uDisparityGlobals->sideLineLeft_m;
			}

			if (previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].xHigh_Index >= Global_uDisparityParams->totalCols)
			{
				previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].xHigh_Index = Global_uDisparityParams->totalCols - 1;
				previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].ylowDisp_Index = round(Global_uDisparityGlobals->sideLineLeft_m*previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].xHigh_Index + Global_uDisparityGlobals->sideLineLeft_b);
			}

			previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].setUdispObject(*Global_uDisparityParams);

			Global_uDisparityGlobals->infinityLeftPointX = previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].xHigh_Index;
			Global_uDisparityGlobals->infinityLeftPointY = previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].ylowDisp_Index;

			Global_dDisparityGlobals->infinityUlineLeftPointX = Global_uDisparityGlobals->infinityLeftPointX;
			Global_dDisparityGlobals->infinityUlineLeftPointY = get_v_From_VDisparity_RoadProfile(Global_uDisparityGlobals->infinityLeftPointY); //round(Global_uDisparityGlobals->infinityLeftPointY*Global_vDisparityGlobals->bestRoadLine_m + Global_vDisparityGlobals->bestRoadLine_b);			
		}

		if (Global_uDisparityGlobals->sideLineRightIndex != -1)
		{
			//set rightleaning line lower disp
			previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].isSideBuilding = true;
			Global_uDisparityGlobals->sideLineRight_m = previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].udisp_slope;
			Global_uDisparityGlobals->sideLineRight_b = previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].udisp_slope_b;

			if (previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].ylowDisp_Index > 4)
			{
				previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].ylowDisp_Index = 4;
				previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].xLow_Index = (previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].ylowDisp_Index - Global_uDisparityGlobals->sideLineRight_b) / Global_uDisparityGlobals->sideLineRight_m;
			}

			if (previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].xLow_Index < 0)
			{
				previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].xLow_Index = 0;
				previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].ylowDisp_Index = round(Global_uDisparityGlobals->sideLineRight_m*previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].xLow_Index + Global_uDisparityGlobals->sideLineRight_b);
			}

			previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].setUdispObject(*Global_uDisparityParams);

			Global_uDisparityGlobals->infinityRightPointX = previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].xLow_Index;
			Global_uDisparityGlobals->infinityRightPointY = previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].ylowDisp_Index;

			Global_dDisparityGlobals->infinityUlineRightPointX = Global_uDisparityGlobals->infinityRightPointX;
			Global_dDisparityGlobals->infinityUlineRightPointY = get_v_From_VDisparity_RoadProfile(Global_uDisparityGlobals->infinityRightPointY); //round(Global_uDisparityGlobals->infinityRightPointY*Global_vDisparityGlobals->bestRoadLine_m + Global_vDisparityGlobals->bestRoadLine_b);
		}
	}

	//cv::Mat uDispDraw2 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, CV_8UC3);
	//uDispDraw2 = drawUDispObjects(uDispDraw2, previousObjectList);

	if (Global_uDisparityGlobals->sideLineLeftIndex != -1 || Global_uDisparityGlobals->sideLineRightIndex != -1)
	{
		//Filtering
		for (int obji = 0; obji < Global_currentNumOfObjects; obji++)
		{
			obj = previousObjectList[obji];

			if (!obj.isSideBuilding)
			{
				//Instead of using mid points (test it properly), try far point, delete object if its far point behind the walls
				float objMidPointX = 0, objMidPointY = 0;

				if (obj.isLeftLeaning) {
					objMidPointX = obj.xHigh_Index;//since xHigh_Index is the far point in this case
				}
				else {
					objMidPointX = obj.xLow_Index; //since xlowIndex is the far point in this case
				}

				objMidPointY = round(objMidPointX*obj.udisp_slope + obj.udisp_slope_b);

				if (Global_uDisparityGlobals->sideLineLeftIndex != -1)
				{
					float objOnLeftLineYLeft = round(objMidPointX*Global_uDisparityGlobals->sideLineLeft_m + Global_uDisparityGlobals->sideLineLeft_b);
					objMidPointX = round((obj.xHigh_Index + obj.xLow_Index) / 2.0);
					if ((objMidPointX > previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].xLow_Index && objMidPointX < previousObjectList[Global_uDisparityGlobals->sideLineLeftIndex].xHigh_Index) && objMidPointY <= objOnLeftLineYLeft)
					{
						previousObjectList[obji].isDeleted = true;
					}
				}

				if (Global_uDisparityGlobals->sideLineRightIndex != -1)
				{
					float objOnRightLineYRight = round(objMidPointX*Global_uDisparityGlobals->sideLineRight_m + Global_uDisparityGlobals->sideLineRight_b);
					objMidPointX = round((obj.xHigh_Index + obj.xLow_Index) / 2.0);
					if ((objMidPointX > previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].xLow_Index && objMidPointX < previousObjectList[Global_uDisparityGlobals->sideLineRightIndex].xHigh_Index) && objMidPointY <= objOnRightLineYRight)
					{
						previousObjectList[obji].isDeleted = true;
					}
				}
			}
		}
	}


	//cv::Mat uDispDraw3 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, CV_8UC3);
	//uDispDraw3 = drawUDispObjects(uDispDraw3, previousObjectList);
	//cout << "hey" << endl;
}

void inline StereoImageBuffer::createDenseDispObjects(std::vector<MyObject, NAlloc<MyObject>>& uDispObjectList)
{
	//cv::Mat colored_Disparity;
	//cv::cvtColor(Global_Current_DenseDisparity_Image, colored_Disparity, CV_GRAY2RGB);

	//cv::Mat originalIMGLeft = Global_Current_Left_Image;
	//cv::Mat originalIMG = Global_Current_Left_Image;

	//cv::Mat vDisparity = Global_vDisparityGlobals->vDisparityImage;
	//cv::Mat uDisparity = Global_uDisparityGlobals->uDisparityImage;
	//cv::Mat uDisparity_thresholded = Global_uDisparityGlobals->uDisparityImage_thresholded;

	cv::Mat colorV;
	cv::cvtColor(Global_vDisparityGlobals->vDisparityImage, colorV, CV_GRAY2BGR);
	//cv::line(colorV, Point(0, Global_vDisparityGlobals->bestRoadLine_vIntersection), Point(Global_vDisparityGlobals->bestRoadLine_dInterSection, Global_vDisparityParams->totalRows - 1), cv::Scalar(0, 0, 255));
	//cv::line(colorV, Point(Global_vDisparityGlobals->piecewiseBestLine[0], Global_vDisparityGlobals->piecewiseBestLine[1]), Point(Global_vDisparityGlobals->piecewiseBestLine[2], Global_vDisparityGlobals->piecewiseBestLine[3]), cv::Scalar(0, 255, 0));

	cv::Mat uDispDraw = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, colorV.type());
	uDispDraw = drawUDispObjects(uDispDraw, uDispObjectList);

	int bestLine_vMax = Global_vDisparityParams->totalRows - 1;
	for (size_t i = 0; i < Global_currentNumOfObjects; i++)
	{
		if (!uDispObjectList[i].isDeleted)
		{

			//int vMax = (b - dMax) / m; // vMax = ymax (iloi), bottom line of the object	
			if (uDispObjectList[i].yhighDisp_Index > Global_vDisparityGlobals->bestRoadLine_dInterSection)
			{
				uDispObjectList[i].yhighDisp_Index = Global_vDisparityGlobals->bestRoadLine_dInterSection;
				uDispObjectList[i].setUdispObject(*Global_uDisparityParams);
			}

			if (uDispObjectList[i].ylowDisp_Index > Global_vDisparityGlobals->bestRoadLine_dInterSection)
			{
				uDispObjectList[i].ylowDisp_Index = Global_vDisparityGlobals->bestRoadLine_dInterSection;
				uDispObjectList[i].setUdispObject(*Global_uDisparityParams);
			}

			uDispObjectList[i].yhighDisp_Vmax_Index = get_v_From_VDisparity_RoadProfile(uDispObjectList[i].yhighDisp_Index); //uDispObjectList[i].yhighDisp_Index*Global_vDisparityGlobals->bestRoadLine_m + Global_vDisparityGlobals->bestRoadLine_b; // vMax = ymax (iloi), bottom line of the object
			uDispObjectList[i].ylowDisp_Vmax_Index = get_v_From_VDisparity_RoadProfile(uDispObjectList[i].ylowDisp_Index); //uDispObjectList[i].ylowDisp_Index*Global_vDisparityGlobals->bestRoadLine_m + Global_vDisparityGlobals->bestRoadLine_b;
			uDispObjectList[i].set_dDispObjLine();


			if (uDispObjectList[i].yhighDisp_Vmax_Index >= 0)//else it's out of range (not on the road)
			{
				int objBasePoint = uDispObjectList[i].yhighDisp_Vmax_Index;
				if (objBasePoint <= Global_vDisparityGlobals->bestRoadLine_vIntersection)//if the object base is above the horizontal delete it (e.g birds)
				{
					uDispObjectList[i].isDeleted = true;
					//cout << "obj:" << uDispObjectList[i].labelNum << " index:" << i << " will be deleted due to object base is above the horizontal" << endl;
				}
				else
				{
					if (!uDispObjectList[i].isSideBuilding)
					{
						get_Upperline_From_denseDisparity(uDispObjectList[i]);

						if (uDispObjectList[i].objSize < Global_dDisparityParams->minObjSizeAllowed || (uDispObjectList[i].ylowDisp_Index <= Global_uDisparityParams->maxDOfSideBuildings && uDispObjectList[i].objSize> Global_dDisparityParams->maxObjSizeAllowed && uDispObjectList[i].objHeight < Global_dDisparityParams->minObjHeightAllowed))
						{
							uDispObjectList[i].isDeleted = true;
							//cout << "obj:" << uDispObjectList[i].labelNum << " index:" << i << " will be deleted due to minObjSizeAllowed" << endl;
							continue;
						}

						//delete object if vmax and vmin almost intersects
						int checkVminVmaxValue = abs(uDispObjectList[i].ylowDisp_Vmax_Index - uDispObjectList[i].ylowDisp_Vmin_Index);

						if (checkVminVmaxValue < Global_dDisparityParams->minVminVmaxDifDeleteThreshold)
						{
							uDispObjectList[i].isDeleted = true;
							//cout << "obj:" << uDispObjectList[i].labelNum << " index:" << i << " will be deleted due to minVminVmaxDifDeleteThreshold" << endl;
						}
					}
					else
					{
						uDispObjectList[i].setDensDispObject(true, Global_Current_DenseDisparity_Image32F);
					}
				}
			}
			else
			{
				uDispObjectList[i].isDeleted = true;
				//cout << "obj:" << uDispObjectList[i].labelNum << " index:" << i << " will be deleted due to yhighDisp_Vmax_Index < 0" << endl;

			}
		}
	}

	//cv::Mat uDispDraw2 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, colorV.type());
	//uDispDraw2 = drawUDispObjects(uDispDraw2, uDispObjectList);

	//for (size_t i = 0; i < Global_currentNumOfObjects; i++)
	//{
	//	if (!uDispObjectList[i].isDeleted)
	//	{
	//		printObject(uDispObjectList[i]);
	//	}
	//}	

	//cv::Mat drawedDense = drawObjects(Global_Current_Left_Image, Global_mergedObjectList2);
	//cout << "hey" << endl;
}

void inline StereoImageBuffer::get_Upperline_From_denseDisparity(MyObject& obj)
{
	const unsigned char* row_disparity;
	bool stop = false;
	float avg_dense = 0;
	int dMin = obj.ylowDisp_Index;
	int dMax = obj.yhighDisp_Index;
	float base_threshold = round(dMax - (dMax * Global_dDisparityParams->upperline_extendDisparitySearchConst));
	float up_threshold = round(dMax + (dMax * Global_dDisparityParams->upperline_extendDisparitySearchConst));
	int x_left_Index = 0, x_right_Index = 0;
	int basePointV = obj.yhighDisp_Vmax_Index;
	int counter = 0;
	int nDisparity = dMax - dMin + 1;
	int rangeDisparity = nDisparity / 4 + 1;
	//cout << "Disp:" << nDisparity << " - " << " range:" << rangeDisparity << endl;
	bool isStraightline = false;

	//cv::Mat colored_Disparity;
	//cv::cvtColor(Global_Current_DenseDisparity_Image, colored_Disparity, CV_GRAY2RGB);

	//cv::Mat originalIMG = Global_Current_Left_Image;

	//cv::Mat uDispDraw2 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, CV_8UC3);
	//uDispDraw2 = drawUDispObjects(uDispDraw2, Global_mergedObjectList2);

	int ignoreErrorThresholdD = round(Global_numDisparities - (Global_numDisparities / 4));

	if (abs(obj.ddisp_slope) > 0)
	{
		if (obj.isLeftLeaning)
		{
			x_left_Index = (dMax - obj.udisp_slope_b) / obj.udisp_slope;
			x_right_Index = ((dMax - rangeDisparity) - obj.udisp_slope_b) / obj.udisp_slope;
		}
		else
		{
			x_left_Index = ((dMax - rangeDisparity) - obj.udisp_slope_b) / obj.udisp_slope;
			x_right_Index = (dMax - obj.udisp_slope_b) / obj.udisp_slope;
		}
	}
	else //if object is straight line
	{
		x_left_Index = obj.xLow_Index;
		x_right_Index = obj.xHigh_Index;
		isStraightline = true;
	}

	int v = basePointV;
	while (!stop && v>0)
	{
		v = v - 1;

		row_disparity = Global_Current_DenseDisparity_Image.ptr<const unsigned char>(v);
		avg_dense = 0;
		counter = 0;
		for (int u = x_left_Index; u <= x_right_Index; u++)
		{
			int d = row_disparity[u];
			if (((d >= base_threshold) && (d <= up_threshold)) || ((dMax > ignoreErrorThresholdD) && (d<2)) || (d<1)) //ignore other points until no d aprx dMax is right
			{
				counter += 1;
				avg_dense = avg_dense + d;
			}

		}
		avg_dense = avg_dense / counter;
		int counterThreshold = ((x_right_Index - x_left_Index + 1) / Global_dDisparityParams->upperline_minCounterThresholdConst) + 1;

		if (/*(avg_dense < base_threshold) || (avg_dense > up_threshold) ||*/ counter < counterThreshold)
		{
			stop = true;
			break;
		}
	}

	obj.yhighDisp_Vmin_Index = v;
	obj.setDensDispObject(true, Global_Current_DenseDisparity_Image32F);

	if (((obj.ylowDisp_Vmax_Index - obj.ylowDisp_Vmin_Index) <= Global_dDisparityParams->upperline_minVminVmaxDifThreshold) && !isStraightline)
	{
		base_threshold = round(dMin - (dMin * Global_dDisparityParams->upperline_extendDisparitySearchConst));
		up_threshold = round(dMin + (dMin * Global_dDisparityParams->upperline_extendDisparitySearchConst));
		basePointV = obj.ylowDisp_Vmax_Index;

		if (obj.isLeftLeaning)
		{
			x_left_Index = ((dMin + rangeDisparity) - obj.udisp_slope_b) / obj.udisp_slope;
			x_right_Index = (dMin - obj.udisp_slope_b) / obj.udisp_slope;
		}
		else
		{
			x_left_Index = (dMin - obj.udisp_slope_b) / obj.udisp_slope;
			x_right_Index = ((dMin + rangeDisparity) - obj.udisp_slope_b) / obj.udisp_slope;
		}

		v = basePointV;
		stop = false;
		while (!stop && v>0)
		{
			v = v - 1;

			row_disparity = Global_Current_DenseDisparity_Image.ptr<const unsigned char>(v);
			avg_dense = 0;
			counter = 0;
			for (int u = x_left_Index; u <= x_right_Index; u++)
			{
				int d = row_disparity[u];

				if (((d >= base_threshold) && (d <= up_threshold)) || ((dMax > ignoreErrorThresholdD) && (d<2)) || (d<1)) //ignore other points until no d aprx dMax is right
				{
					counter += 1;
					avg_dense = avg_dense + d;
				}

			}
			avg_dense = avg_dense / counter;
			int counterThreshold = ((x_right_Index - x_left_Index + 1) / Global_dDisparityParams->upperline_minCounterThresholdConst) + 1;

			if (/*(avg_dense < base_threshold) || (avg_dense > up_threshold) ||*/ counter < counterThreshold)
			{
				stop = true;
				break;
			}
		}

		obj.ylowDisp_Vmin_Index = v;
		obj.setDensDispObject(false, Global_Current_DenseDisparity_Image32F);
	}
}

inline cv::Mat StereoImageBuffer::drawUDispObjects(cv::Mat uDisp, std::vector<MyObject, NAlloc<MyObject>>& objects/*std::vector<MyObject> objects*/) const
{
	std::ostringstream text;

	if (Global_currentNumOfObjects > 0)
	{
		//draw main frame
		cv::line(uDisp, cv::Point(Global_uDisparityParams->mainFrameLeftCol, 0), Point(Global_uDisparityParams->mainFrameLeftCol, Global_uDisparityParams->totalRows), cv::Scalar(255, 255, 255), 1);
		cv::line(uDisp, cv::Point(Global_uDisparityParams->mainFrameRightCol, 0), Point(Global_uDisparityParams->mainFrameRightCol, Global_uDisparityParams->totalRows), cv::Scalar(255, 255, 255), 1);

		//draw middle
		int col = Global_uDisparityParams->middleCol - 1;
		int scalarVar;
		int i;
		for (i = 0; i <= Global_uDisparityParams->zeroD; i = i + Global_uDisparityParams->zeroD_MaskV)
		{
			scalarVar = (i*i*i * 4) % 255;
			cv::line(uDisp, cv::Point(col, i), Point(col, i + Global_uDisparityParams->zeroD_MaskV - 1), cv::Scalar(scalarVar, scalarVar, 255), 1);
		}

		for (i = i; i <= Global_uDisparityParams->firstQuarterD; i = i + Global_uDisparityParams->quarterD_MaskV)
		{
			scalarVar = (i*i*i * 3) % 255;
			cv::line(uDisp, cv::Point(col, i), Point(col, i + Global_uDisparityParams->quarterD_MaskV - 1), cv::Scalar(255, scalarVar, scalarVar), 1);
			cv::line(uDisp, cv::Point(col - 1, i), Point(col - 1, i + Global_uDisparityParams->quarterD_MaskV - 1), cv::Scalar(255, scalarVar, scalarVar), 1);
		}

		for (i = i; i <= Global_uDisparityParams->secondQuarterD; i = i + Global_uDisparityParams->halfD_MaskV)
		{
			scalarVar = (i*i*i * 2) % 255;
			cv::line(uDisp, cv::Point(col, i), Point(col, i + Global_uDisparityParams->halfD_MaskV - 1), cv::Scalar(scalarVar, 255, scalarVar), 1);
		}

		for (i = i; i <= Global_uDisparityParams->thirdQuarterD; i = i + Global_uDisparityParams->threequarterD_MaskV)
		{
			scalarVar = (i*i*i) % 255;
			cv::line(uDisp, cv::Point(col, i), Point(col, i + Global_uDisparityParams->threequarterD_MaskV - 1), cv::Scalar(scalarVar, scalarVar, 255), 1);
			cv::line(uDisp, cv::Point(col - 1, i), Point(col - 1, i + Global_uDisparityParams->threequarterD_MaskV - 1), cv::Scalar(scalarVar, scalarVar, 255), 1);
		}

		for (i = i; i < Global_uDisparityParams->maxD; i = i + Global_uDisparityParams->maxD_MaskV)
		{
			scalarVar = (i*i*i) % 255;
			cv::line(uDisp, cv::Point(col, i), Point(col, i + Global_uDisparityParams->maxD_MaskV - 1), cv::Scalar(255, scalarVar, scalarVar), 1);
		}



		cv::line(uDisp, cv::Point(Global_uDisparityParams->middleCol, 0), Point(Global_uDisparityParams->middleCol, Global_uDisparityParams->totalRows), cv::Scalar(255, 255, 255), 1);

		for (size_t obj_index = 0; obj_index < Global_currentNumOfObjects; obj_index++)
		{
			MyObject obj = objects[obj_index];

			int h = obj.getUdisp_Height();
			int l = obj.getUdisp_Length();
			int d = obj.getAvgDisparity();

			int x1 = obj.xLow_Index;
			int y1 = obj.yhighDisp_Index;
			int x2 = obj.xHigh_Index;
			int y2 = obj.ylowDisp_Index;

			if (obj.isDeleted)
			{
				if (obj.isLeftLeaning)
				{
					cv::line(uDisp, cv::Point(x1, y1), Point(x2, y2), cv::Scalar(255, 255, 255), 1);
				}
				else
				{
					cv::line(uDisp, cv::Point(x1, y2), Point(x2, y1), cv::Scalar(255, 255, 255), 1);
				}
				continue;
			}

			if (obj.isSideBuilding)
			{
				if (obj.isLeftLeaning)
				{
					cv::line(uDisp, cv::Point(x1, y1), Point(x2, y2), cv::Scalar(0, 255, 255), 1);
					cv::circle(uDisp, Point(x2, y2), 2, cv::Scalar(0, 255, 255));
				}
				else
				{
					cv::line(uDisp, cv::Point(x1, y2), Point(x2, y1), cv::Scalar(0, 255, 255), 1);
					cv::circle(uDisp, Point(x1, y2), 2, cv::Scalar(255, 255, 255));
				}

				continue;
			}

			text << obj_index << ":" << y2 << "," << y1; //<< ") H:" << h << " L:" << l << " D:" << d;
			if (d > 0 && l > 0)
			{
				//cv::line(uDisp, obj.getUdisp_LeftlineP1(), obj.getUdisp_LeftlineP2(), cv::Scalar(0, 0, 255), 1);
				//cv::line(uDisp, obj.getUdisp_RightlineP1(), obj.getUdisp_RightlineP2(), cv::Scalar(0, 0, 255), 1);
				//cv::line(uDisp, obj.getUdisp_UplineP1(), obj.getUdisp_UplineP2(), cv::Scalar(0, 0, 255), 1);
				//cv::line(uDisp, obj.getUdisp_BottomlineP1(), obj.getUdisp_BottomlineP2(), cv::Scalar(0, 0, 255), 1);
				int scalar1 = (obj_index * 10) % 255;
				int scalar2 = (d * 25) % 255;

				if (obj.isLeftLeaning)//if obj is located at the left side of the screen
				{
					if (obj_index % 2 == 0)
					{
						cv::line(uDisp, cv::Point(x1, y1), Point(x2, y2), cv::Scalar(0, scalar2, scalar1), 1);
					}
					else
					{
						cv::line(uDisp, cv::Point(x1, y1), Point(x2, y2), cv::Scalar(scalar2, 0, scalar1), 1);
					}

				}
				else//if obj is located at the right side of the screen
				{
					if (obj_index % 2 == 0)
					{
						cv::line(uDisp, cv::Point(x1, y2), Point(x2, y1), cv::Scalar(0, scalar2, scalar1), 1);
					}
					else
					{
						cv::line(uDisp, cv::Point(x1, y2), Point(x2, y1), cv::Scalar(scalar2, 0, scalar1), 1);
					}
				}

				int textPointX, textPointY;
				float thickness;
				if (d > 16)
				{
					textPointX = (x1 + x2) / 2;
					textPointY = y1 + 4;
					thickness = 0.5;
				}
				else
				{
					textPointX = (x1 + x2) / 2;
					textPointY = y1 + 6;
					thickness = 0.35;
				}

				if (obj_index % 2 == 0) putText(uDisp, text.str(), Point(textPointX, textPointY), cv::FONT_HERSHEY_COMPLEX_SMALL, thickness, cv::Scalar(0, scalar2, scalar1), 1, CV_AA);
				else putText(uDisp, text.str(), Point(textPointX, textPointY), cv::FONT_HERSHEY_COMPLEX_SMALL, thickness, cv::Scalar(scalar2, 0, scalar1), 1, CV_AA);

			}
			else
			{
				//putText(leftIMG, text.str(), obj.getRightlineP1(), cv::FONT_HERSHEY_PLAIN, 0.6, cv::Scalar(0, 255, 255), 1, CV_AA);
				//putText(leftIMG, text.str(), obj.getRightlineP2(), cv::FONT_HERSHEY_PLAIN, 0.6, cv::Scalar(0, 255, 255), 1, CV_AA);
				////std://cout << text.str() << std::endl;
			}
			text.str("");
			text.clear();
		}
	}

	return uDisp;
	//cv::imshow("Window1", leftIMG);
	//cvWaitKey(1000);
}

inline cv::Mat StereoImageBuffer::drawUDispObjects3(cv::Mat uDisp, std::vector<MyObject, NAlloc<MyObject>>& objects, int startIndex, int endIndex) const
{
	std::ostringstream text;
	int numOfObjects = endIndex - startIndex + 1;

	if (numOfObjects > 0)
	{
		for (size_t obj_index = startIndex; obj_index <= endIndex; obj_index++)
		{
			MyObject obj = objects[obj_index];
			if (obj.isDeleted) continue;

			int h = obj.getUdisp_Height();
			int l = obj.getUdisp_Length();
			int d = obj.getAvgDisparity();

			int x1 = obj.xLow_Index;
			int y1 = obj.yhighDisp_Index;
			int x2 = obj.xHigh_Index;
			int y2 = obj.ylowDisp_Index;

			if (obj.isSideBuilding)
			{
				if (obj.isLeftLeaning)
				{
					cv::line(uDisp, cv::Point(x1, y1), Point(x2, y2), cv::Scalar(0, 255, 255), 1);
				}
				else
				{
					cv::line(uDisp, cv::Point(x1, y2), Point(x2, y1), cv::Scalar(0, 255, 255), 1);
				}

				continue;
			}

			text << obj_index << ":" << y2 << "," << y1; //<< ") H:" << h << " L:" << l << " D:" << d;
			if (d > 0 && l > 0)
			{
				//cv::line(uDisp, obj.getUdisp_LeftlineP1(), obj.getUdisp_LeftlineP2(), cv::Scalar(0, 0, 255), 1);
				//cv::line(uDisp, obj.getUdisp_RightlineP1(), obj.getUdisp_RightlineP2(), cv::Scalar(0, 0, 255), 1);
				//cv::line(uDisp, obj.getUdisp_UplineP1(), obj.getUdisp_UplineP2(), cv::Scalar(0, 0, 255), 1);
				//cv::line(uDisp, obj.getUdisp_BottomlineP1(), obj.getUdisp_BottomlineP2(), cv::Scalar(0, 0, 255), 1);
				int scalar1 = (obj_index * 10) % 255;
				int scalar2 = (d * 25) % 255;

				if (obj.isLeftLeaning)//if obj is located at the left side of the screen
				{
					if (obj_index % 2 == 0)
					{
						cv::line(uDisp, cv::Point(x1, y1), Point(x2, y2), cv::Scalar(0, scalar2, scalar1), 1);
					}
					else
					{
						cv::line(uDisp, cv::Point(x1, y1), Point(x2, y2), cv::Scalar(scalar2, 0, scalar1), 1);
					}

				}
				else//if obj is located at the right side of the screen
				{
					if (obj_index % 2 == 0)
					{
						cv::line(uDisp, cv::Point(x1, y2), Point(x2, y1), cv::Scalar(0, scalar2, scalar1), 1);
					}
					else
					{
						cv::line(uDisp, cv::Point(x1, y2), Point(x2, y1), cv::Scalar(scalar2, 0, scalar1), 1);
					}
				}

				int textPointX, textPointY;
				float thickness;
				if (d > 16)
				{
					textPointX = (x1 + x2) / 2;
					textPointY = y1 + 4;
					thickness = 0.5;
				}
				else
				{
					textPointX = (x1 + x2) / 2;
					textPointY = y1 + 6;
					thickness = 0.35;
				}

				if (obj_index % 2 == 0) putText(uDisp, text.str(), Point(textPointX, textPointY), cv::FONT_HERSHEY_COMPLEX_SMALL, thickness, cv::Scalar(0, scalar2, scalar1), 1, CV_AA);
				else putText(uDisp, text.str(), Point(textPointX, textPointY), cv::FONT_HERSHEY_COMPLEX_SMALL, thickness, cv::Scalar(scalar2, 0, scalar1), 1, CV_AA);

			}
			else
			{
				//putText(leftIMG, text.str(), obj.getRightlineP1(), cv::FONT_HERSHEY_PLAIN, 0.6, cv::Scalar(0, 255, 255), 1, CV_AA);
				//putText(leftIMG, text.str(), obj.getRightlineP2(), cv::FONT_HERSHEY_PLAIN, 0.6, cv::Scalar(0, 255, 255), 1, CV_AA);
				////std://cout << text.str() << std::endl;
			}
			text.str("");
			text.clear();
		}
	}

	return uDisp;
	//cv::imshow("Window1", leftIMG);
	//cvWaitKey(1000);
}

#ifdef WITH_SLAM
void StereoImageBuffer::addDrawPoint(int imgNum, string txt, int z, int x, int w, int h, Mat & occ, Mat & slam, float realDistance)
{
	x = x * 2;
	z = z * 2;
	w = w * 2;
	h = h * 2;

	cv::Point2i old_point(x, z);
	cv::Point2i old_point2(x + w - 1, z + h - 1);
	cv::rectangle(occ, old_point, old_point2, cv::Scalar(0, 255, 0));
	float closest_distance = 0;
	int closest_pointX = 0;
	int closest_pointZ = 0;

	float max_old = 0;
	float min_dist = INT_MAX;
	for (int v_index = z; v_index < (z + h); v_index++)
	{
		for (int u_index = x; u_index < (x + w); u_index++)
		{
			float Pou = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix.at<float>(v_index, u_index);

			if (Pou > max_old)
				max_old = Pou;

			float zw = -(v_index - Global_occupancyGridParams->totalRows + 1) * Global_occupancyGridParams->ZWorldResolution;
			float xw = (u_index - Global_occupancyGridParams->middleCol) * Global_occupancyGridParams->XWorldResolution;
			float dist = sqrt(pow(xw, 2) + pow(zw, 2));

			float dif_dist = abs(realDistance - dist);
			if (abs(dif_dist < min_dist))
			{
				min_dist = dif_dist;
				closest_distance = dist;
				closest_pointX = u_index;
				closest_pointZ = v_index;
			}
		}
	}

	cv::Point2i new_point = Global_occupancyGridParams->setMapCoordinatesByNN(x, z);
	cv::Point2i new_point2 = Global_occupancyGridParams->setMapCoordinatesByNN(x + w - 1, z + h - 1);
	cv::Point2i c_point = Global_occupancyGridParams->setMapCoordinatesByNN(closest_pointX, closest_pointZ);

	//int z2 = round((old_point.y + old_point2.y) / 2);
	//int x2 = round((old_point.x + old_point2.x) / 2);

	//float zw = -(z2 - Global_occupancyGridParams->totalRows + 1) * Global_occupancyGridParams->ZWorldResolution;
	//float xw = (x2 - Global_occupancyGridParams->middleCol) * Global_occupancyGridParams->XWorldResolution;
	//float dr = sqrt(pow(xw, 2) + pow(zw, 2));



	//cv::rectangle(slam, new_point, new_point2, cv::Scalar(0, 255, 0));

	//cv::putText(slam, txt, new_point, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, cv::Scalar(0, 255, 255), 0.25, CV_AA);


	drawings.push_back(new myDraw(imgNum, txt, closest_distance, max_old, new_point, new_point2, c_point));
}
#endif

inline cv::Mat StereoImageBuffer::drawObjects(cv::Mat drawingImage, std::vector<MyObject, NAlloc<MyObject>>& objects) const
{
	std::ostringstream text;
	std::ostringstream text2;
	std::ostringstream text3;

	cv::Mat colored_drawingImage;
	cv::cvtColor(drawingImage, colored_drawingImage, CV_GRAY2RGB);

	circle(colored_drawingImage, Point(Global_dDisparityGlobals->infinityPointX, Global_vDisparityGlobals->piecewiseBestLine_RoadEndPoint.y), 5, cv::Scalar(255, 255, 0), 3);
	line(colored_drawingImage, Point(0, drawingImage.rows - 1), Point(Global_dDisparityGlobals->infinityPointX, Global_dDisparityGlobals->infinityPointY), cv::Scalar(255, 255, 0), 2);
	line(colored_drawingImage, Point(drawingImage.cols - 1, drawingImage.rows - 1), Point(Global_dDisparityGlobals->infinityPointX, Global_dDisparityGlobals->infinityPointY), cv::Scalar(255, 255, 0), 2);

	//delete later
	for (int id = Global_vDisparityGlobals->piecewiseBestLine_RoadEndPoint.x; id <= Global_vDisparityParams->totalCols - 1; id = id + 1)
	{
		int vmax = get_v_From_VDisparity_RoadProfile(id); //id*Global_vDisparityGlobals->bestRoadLine_m + Global_vDisparityGlobals->bestRoadLine_b;
		int x_left_roadlimit = ((vmax - Global_dDisparityGlobals->vehicleRoadLineLeft_b) / Global_dDisparityGlobals->vehicleRoadLineLeft_m);
		int x_right_roadlimit = ((vmax - Global_dDisparityGlobals->vehicleRoadLineRight_b) / Global_dDisparityGlobals->vehicleRoadLineRight_m);

		cv::line(colored_drawingImage, Point(x_left_roadlimit, vmax), Point(x_right_roadlimit, vmax), Scalar(255, 255, 0));
	}
	//delete later

	//horizon line from vDisparity
	cv::line(colored_drawingImage, Point(0, Global_vDisparityGlobals->bestRoadLine_vIntersection), Point(drawingImage.cols - 1, Global_vDisparityGlobals->bestRoadLine_vIntersection), cv::Scalar(255, 0, 255), 2);

	//float arrowXLen = 20;
	//float arrowYLen = 10;
	////float m = arrowYLen / arrowXLen ;//


	//float r = 13;
	//int difx_left, dify_left, x_left, y_left;
	//int difx_right, dify_right, x_right, y_right;

	//if (Global_dDisparityGlobals->infinityUlineLeftPointY != -1 && Global_dDisparityGlobals->infinityUlineLeftPointX != -1)
	//{
	//	float tetaLeftRad = (135 / 180.0) * CV_PI;
	//	float tetaRightRad = (225 / 180.0) * CV_PI;

	//	float startX = round((drawingImage.cols / 2));
	//	float startY = drawingImage.rows - 1;
	//	float endX = Global_dDisparityGlobals->infinityUlineLeftPointX;
	//	float endY = Global_dDisparityGlobals->infinityUlineLeftPointY;
	//	float m = (endY - startY) / (endX - startX);
	//	float rad = std::atan(m);
	//	float deg = rad*(180 / CV_PI);
	//	tetaLeftRad = rad - tetaLeftRad;
	//	float tetaLeftDeg = tetaLeftRad*(180 / CV_PI);

	//	line(colored_drawingImage, Point(startX, startY), Point(endX, endY), cv::Scalar(0, 255, 255), 2);
	//	
	//	//Make Arrows with polar coordinates --> x = r*cos(teta)  y = r*sin(teta) teta = atan(y/x) r = sqrt(sqr(x) + sqr(y))
	//	difx_left = round(r*cos(tetaLeftRad));
	//	dify_left = round(r*sin(tetaLeftRad));

	//	x_left = endX + difx_left;
	//	y_left = endY + dify_left;

	//	line(colored_drawingImage, Point(endX, endY), Point(x_left, y_left), cv::Scalar(0, 255, 255), 2);

	//	tetaRightRad = rad - tetaRightRad;
	//	float tetaRightDeg = tetaRightRad*(180 / CV_PI);
	//	difx_right = round(r*cos(tetaRightRad));
	//	dify_right = round(r*sin(tetaRightRad));

	//	x_right = endX + difx_right;
	//	y_right = endY + dify_right;

	//	line(colored_drawingImage, Point(endX, endY), Point(x_right, y_right), cv::Scalar(0, 255, 255), 2);
	//}
	//
	//if (Global_dDisparityGlobals->infinityUlineRightPointY != -1 && Global_dDisparityGlobals->infinityUlineRightPointX != -1)
	//{
	//	float tetaLeftRad = (135 / 180.0) * CV_PI;
	//	float tetaRightRad = (225 / 180.0) * CV_PI;

	//	float startX = round((drawingImage.cols / 2));
	//	float startY = drawingImage.rows - 1;
	//	float endX = Global_dDisparityGlobals->infinityUlineRightPointX;
	//	float endY = Global_dDisparityGlobals->infinityUlineRightPointY;
	//	float m = (endY - startY) / (endX - startX);
	//	float rad = std::atan(m);
	//	float deg = rad*(180 / CV_PI);
	//	tetaLeftRad = tetaLeftRad - rad;
	//	float tetaLeftDeg = tetaLeftRad*(180 / CV_PI);

	//	line(colored_drawingImage, Point(startX, startY), Point(endX, endY), cv::Scalar(0, 255, 255), 2);

	//	//Make Arrows with polar coordinates --> x = r*cos(teta)  y = r*sin(teta) teta = atan(y/x) r = sqrt(sqr(x) + sqr(y))
	//	difx_left = round(r*cos(tetaLeftRad));
	//	dify_left = round(r*sin(tetaLeftRad));

	//	x_left = endX - difx_left;
	//	y_left = endY + dify_left;

	//	line(colored_drawingImage, Point(endX, endY), Point(x_left, y_left), cv::Scalar(0, 255, 255), 2);

	//	tetaRightRad = tetaRightRad;
	//	float tetaRightDeg = tetaRightRad*(180 / CV_PI);
	//	difx_right = round(r*cos(tetaRightRad));
	//	dify_right = round(r*sin(tetaRightRad));

	//	x_right = endX - difx_right;
	//	y_right = endY + dify_right;

	//	line(colored_drawingImage, Point(endX, endY), Point(x_right, y_right), cv::Scalar(0, 255, 255), 2);
	//}

	if (Global_dDisparityGlobals->infinityUlineCommonPointY != -1 && Global_dDisparityGlobals->infinityUlineCommonPointX != -1)
	{
		circle(colored_drawingImage, Point(Global_dDisparityGlobals->infinityUlineCommonPointX, Global_dDisparityGlobals->infinityUlineCommonPointY), 5, cv::Scalar(0, 255, 255), 3);
		line(colored_drawingImage, Point(0, drawingImage.rows - 1), Point(Global_dDisparityGlobals->infinityUlineCommonPointX, Global_dDisparityGlobals->infinityUlineCommonPointY), cv::Scalar(0, 255, 255), 2);
		line(colored_drawingImage, Point(drawingImage.cols - 1, drawingImage.rows - 1), Point(Global_dDisparityGlobals->infinityUlineCommonPointX, Global_dDisparityGlobals->infinityUlineCommonPointY), cv::Scalar(0, 255, 255), 2);
	}


	int closestObjectIndex = -1;
	float closestObjectDisparity = 1000;

	if (Global_currentNumOfObjects > 0)
	{
		for (size_t obj_index = 0; obj_index < Global_currentNumOfObjects; obj_index++)
		{
			MyObject obj = objects[obj_index];
			if (objects[obj_index].isDeleted)
			{
				if (objects[obj_index].isLeftLeaning)
				{
					cv::line(colored_drawingImage, obj.dDisp_lineUpP1, obj.dDisp_lineBottomP2, cv::Scalar(255, 255, 255), 1);
					cv::line(colored_drawingImage, obj.dDisp_lineUpP2, obj.dDisp_lineBottomP1, cv::Scalar(255, 255, 255), 1);
				}
				else
				{
					cv::line(colored_drawingImage, obj.dDisp_lineUpP1, obj.dDisp_lineBottomP2, cv::Scalar(255, 255, 255), 1);
					cv::line(colored_drawingImage, obj.dDisp_lineUpP2, obj.dDisp_lineBottomP1, cv::Scalar(255, 255, 255), 1);
				}

				cv::line(colored_drawingImage, obj.dDisp_lineLeftP1, obj.dDisp_lineLeftP2, cv::Scalar(255, 255, 255), 1);
				cv::line(colored_drawingImage, obj.dDisp_lineRightP1, obj.dDisp_lineRightP2, cv::Scalar(255, 255, 255), 1);
				cv::line(colored_drawingImage, obj.dDisp_lineUpP1, obj.dDisp_lineUpP2, cv::Scalar(255, 255, 255), 1);
				cv::line(colored_drawingImage, obj.dDisp_lineBottomP1, obj.dDisp_lineBottomP2, cv::Scalar(255, 255, 255), 1);
				continue;
			}

			int h = obj.getDensdisp_Height();
			int l = obj.getDensdisp_Length();
			int d = obj.ylowDisp_Index;
			int basePoint = obj.dDisp_lineBottomP1.y; // = obj.yhighDisp_Vmax_Index; // obj.getDensdisp_BottomlineP1().y;


													  //text << obj_index << ") H:" << h << " L:" << l << " D:" << d;
			putText(colored_drawingImage, getCurrentImageNum().str(), cv::Point(15, 15), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0), 1, CV_AA);

			bool isObjectOnTheRoad = checkIfObjectIsOnTheRoad(obj);

			if (isObjectOnTheRoad)
			{
				if (obj.realDistanceTiltCorrected < closestObjectDisparity)
				{
					closestObjectDisparity = obj.realDistanceTiltCorrected;
					closestObjectIndex = obj_index;
				}

				//make it blue
				cv::line(colored_drawingImage, obj.dDisp_lineLeftP1, obj.dDisp_lineLeftP2, cv::Scalar(255, 0, 0), 2);
				cv::line(colored_drawingImage, obj.dDisp_lineRightP1, obj.dDisp_lineRightP2, cv::Scalar(255, 0, 0), 2);
				cv::line(colored_drawingImage, obj.dDisp_lineUpP1, obj.dDisp_lineUpP2, cv::Scalar(255, 0, 0), 2);
				cv::line(colored_drawingImage, obj.dDisp_lineBottomP1, obj.dDisp_lineBottomP2, cv::Scalar(255, 0, 0), 2);

				int x1 = obj.dDisp_lineLeftP1.x;
				int x2 = obj.dDisp_lineRightP1.x;
				int y1 = obj.dDisp_lineBottomP1.y;

				int textPointX = ((x1 + x2) / 2) - 10;
				//int textPointY = ((y1 + y2) / 2);
				int textPointY = y1 + (obj.ylowDisp_Index / 20) + 2;
				text << fixed << std::setprecision(2) << "Rd:" << obj.realDistanceTiltCorrected << "m. (Dd:" << obj.directDistance << "m.)";
				putText(colored_drawingImage, text.str(), Point(textPointX, textPointY), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 255, 255), 1.5, CV_AA);
				text2 << fixed << std::setprecision(2) << "Xw:" << obj.Xw << " H:" << obj.objHeight << " L:" << obj.objWidth;// << " Size:" << objToTrack.objSize;
				putText(colored_drawingImage, text2.str(), Point(textPointX, textPointY + 14), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 255, 255), 1.5, CV_AA);

				//cout << "obj:" << objects[obj_index].labelNum << " index:" << obj_index << " is on the road" << endl;
			}
			else
			{
				if (!obj.isSideBuilding)//make it green
				{
					cv::line(colored_drawingImage, obj.dDisp_lineLeftP1, obj.dDisp_lineLeftP2, cv::Scalar(0, 255, 0), 2);
					cv::line(colored_drawingImage, obj.dDisp_lineRightP1, obj.dDisp_lineRightP2, cv::Scalar(0, 255, 0), 2);
					cv::line(colored_drawingImage, obj.dDisp_lineUpP1, obj.dDisp_lineUpP2, cv::Scalar(0, 255, 0), 2);
					cv::line(colored_drawingImage, obj.dDisp_lineBottomP1, obj.dDisp_lineBottomP2, cv::Scalar(0, 255, 0), 2);
				}
				else//make it yellow
				{
					cv::line(colored_drawingImage, obj.dDisp_lineLeftP1, obj.dDisp_lineLeftP2, cv::Scalar(0, 255, 255), 2);
					cv::line(colored_drawingImage, obj.dDisp_lineRightP1, obj.dDisp_lineRightP2, cv::Scalar(0, 255, 255), 2);
					cv::line(colored_drawingImage, obj.dDisp_lineUpP1, obj.dDisp_lineUpP2, cv::Scalar(0, 255, 255), 2);
					cv::line(colored_drawingImage, obj.dDisp_lineBottomP1, obj.dDisp_lineBottomP2, cv::Scalar(0, 255, 255), 2);
				}
			}
			text.str("");
			text.clear();

			text2.str("");
			text2.clear();
		}

		if (closestObjectIndex != -1)
		{
			MyObject objToTrack = objects[closestObjectIndex];//make it red
			cv::line(colored_drawingImage, objToTrack.dDisp_lineLeftP1, objToTrack.dDisp_lineLeftP2, cv::Scalar(0, 0, 255), 2);
			cv::line(colored_drawingImage, objToTrack.dDisp_lineRightP1, objToTrack.dDisp_lineRightP2, cv::Scalar(0, 0, 255), 2);
			cv::line(colored_drawingImage, objToTrack.dDisp_lineUpP1, objToTrack.dDisp_lineUpP2, cv::Scalar(0, 0, 255), 2);
			cv::line(colored_drawingImage, objToTrack.dDisp_lineBottomP1, objToTrack.dDisp_lineBottomP2, cv::Scalar(0, 0, 255), 2);

			int x1 = objToTrack.dDisp_lineLeftP1.x;
			int x2 = objToTrack.dDisp_lineRightP1.x;
			int y1 = objToTrack.dDisp_lineBottomP1.y;
			//int y2 = objToTrack.dDisp_lineUpP1.y;

			int textPointX = ((x1 + x2) / 2) - 10;
			//int textPointY = ((y1 + y2) / 2);
			int textPointY = y1 + (objToTrack.ylowDisp_Index / 20) + 2;
			text << fixed << std::setprecision(2) << "Rd:" << objToTrack.realDistanceTiltCorrected << "m. (Dd:" << objToTrack.directDistance << "m.)";
			putText(colored_drawingImage, text.str(), Point(textPointX, textPointY), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 255, 255), 1.5, CV_AA);
			text2 << fixed << std::setprecision(2) << "Xw:" << objToTrack.Xw << " H:" << objToTrack.objHeight << " L:" << objToTrack.objWidth;// << " Size:" << objToTrack.objSize;
			putText(colored_drawingImage, text2.str(), Point(textPointX, textPointY + 14), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 255, 255), 1.5, CV_AA);
		}


	}

	return colored_drawingImage;
}

inline bool StereoImageBuffer::checkIfObjectIsOnTheRoad(MyObject obj) const
{
	if (obj.isSideBuilding) return false;

	int x_left_roadlimit_yHigh = ((obj.yhighDisp_Vmax_Index - Global_dDisparityGlobals->vehicleRoadLineLeft_b) / Global_dDisparityGlobals->vehicleRoadLineLeft_m) - 1;
	int x_right_roadlimit_yHigh = ((obj.yhighDisp_Vmax_Index - Global_dDisparityGlobals->vehicleRoadLineRight_b) / Global_dDisparityGlobals->vehicleRoadLineRight_m) + 1;

	int x_left_roadlimit_yLow = ((obj.ylowDisp_Vmax_Index - Global_dDisparityGlobals->vehicleRoadLineLeft_b) / Global_dDisparityGlobals->vehicleRoadLineLeft_m) - 1;
	int x_right_roadlimit_yLow = ((obj.ylowDisp_Vmax_Index - Global_dDisparityGlobals->vehicleRoadLineRight_b) / Global_dDisparityGlobals->vehicleRoadLineRight_m) + 1;
	int obj_bottom_vx = -1, obj_upper_vx = -1;

	if (obj.isLeftLeaning)
	{
		obj_bottom_vx = obj.xLow_Index;
		obj_upper_vx = obj.xHigh_Index;
	}
	else
	{
		obj_bottom_vx = obj.xHigh_Index;
		obj_upper_vx = obj.xLow_Index;
	}

	bool isObjectOnTheRoad = false;
	//if ((x_left_roadlimit <= obj.xHigh_Index && obj.xHigh_Index <= x_right_roadlimit) || (x_left_roadlimit <= obj.xLow_Index && obj.xLow_Index <= x_right_roadlimit) || (obj.xLow_Index <= x_left_roadlimit && x_left_roadlimit <= obj.xHigh_Index) || (obj.xLow_Index <= x_right_roadlimit && x_right_roadlimit <= obj.xHigh_Index))
	if ((x_left_roadlimit_yHigh <= obj_bottom_vx && obj_bottom_vx <= x_right_roadlimit_yHigh) || (x_left_roadlimit_yLow <= obj_upper_vx && obj_upper_vx <= x_right_roadlimit_yLow))
	{
		isObjectOnTheRoad = true;
	}

	return isObjectOnTheRoad;
}

inline int StereoImageBuffer::get_d_From_VDisparity_RoadProfile(float v) const//could return "-1" if there are no solutions
{
	float d_max = (float)Global_vDisparityParams->totalCols - 1;
	float d;

	if (Global_vDisparityGlobals->bestRoadLine_a == 0)
	{
		d = (v - Global_vDisparityGlobals->bestRoadLine_b) / Global_vDisparityGlobals->bestRoadLine_m;
	}
	else//it is a second order polynom
	{
		float delta = pow(Global_vDisparityGlobals->bestRoadLine_m, 2) - (4 * Global_vDisparityGlobals->bestRoadLine_a*(Global_vDisparityGlobals->bestRoadLine_b - v));//delta = b^2 - 4ac (y =0), b = b-v (y =v), (ax2 + bx + c = 0) = (ax2 + mx + b) (delta = m^2 - 4ab)
		float x1, x2;

		if (delta == 0) //x1 == x2
		{
			d = (-Global_vDisparityGlobals->bestRoadLine_m) / (2 * Global_vDisparityGlobals->bestRoadLine_a);//-b/2a
		}
		else
		{
			if (delta > 0)// there are 2 roots x1, x2
			{
				x1 = ((-Global_vDisparityGlobals->bestRoadLine_m) + sqrt(delta)) / (2 * Global_vDisparityGlobals->bestRoadLine_a);//b = b-v (y = v) // (-b + delta / 2a)
				x2 = ((-Global_vDisparityGlobals->bestRoadLine_m) - sqrt(delta)) / (2 * Global_vDisparityGlobals->bestRoadLine_a);//b = b-v (y = v) // (-b - delta / 2a)

				if (x1 >= 0 && x1 <= d_max)
				{
					d = x1;
				}
				else// x1 is out of the road profile
				{
					if (x2 >= 0 && x2 <= d_max)
					{
						d = x2;
					}
					else // both x1 and x2 is out of the road profile
					{
						if (x1 > d_max || x2 > d_max)
						{
							d = d_max;
						}
						else//must be below zero
						{
							d = d_max; //d = 0; ???? check it later
						}
					}
				}
			}
			else// if(delta < 0) there are no roots, send the local minima/maxima
			{
				d = (-Global_vDisparityGlobals->bestRoadLine_m) / (2 * Global_vDisparityGlobals->bestRoadLine_a);//-b/2a
			}
		}
	}

	int d_int = (int)round(d);

	if (d_int > (int)d_max)
	{
		d_int = (int)d_max;
	}
	else
	{
		if (d_int < 0)
			d_int = 0;
	}

	return d_int;
}

inline int StereoImageBuffer::get_v_From_VDisparity_RoadProfile(float d) const
{
	//float v;
	int v_max = Global_dDisparityParams->totalRows - 1;

	//if (Global_vDisparityGlobals->bestRoadLine_a == 0)
	//{
	//	v = d*Global_vDisparityGlobals->bestRoadLine_m + Global_vDisparityGlobals->bestRoadLine_b;
	//}
	//else
	//{
	//	v = d*Global_vDisparityGlobals->bestRoadLine_m + pow(d,2)*Global_vDisparityGlobals->bestRoadLine_a + Global_vDisparityGlobals->bestRoadLine_b;
	//}

	//int v_int = (int)round(v);			
	//
	//if (v_int > v_max)
	//{
	//	v_int = v_max;
	//}
	//else
	//{
	//	if (v_int < 0)
	//		v_int = 0;
	//}

	int v_int = 0;
	get_v_From_Calibration(d, 0.0, v_int);

	if (v_int > v_max)
	{
		v_int = v_max;
	}
	else
	{
		if (v_int < 0)
			v_int = 0;
	}

	return v_int;
}

inline double StereoImageBuffer::getPI() const
{
	return PI;
}

inline ostringstream StereoImageBuffer::getCurrentImageNum() const
{
	int Global_currentIMGNum = Global_minIMGIndex + Global_currentIMGIndex * Global_imgIncrement;

	ostringstream text;
	text << "Beta: " << Global_currentIMGNum;

	return text;
}

inline int StereoImageBuffer::getCurrentImageNumInt() const
{
	int Global_currentIMGNum = Global_minIMGIndex + Global_currentIMGIndex * Global_imgIncrement;
	return Global_currentIMGNum;
}

static struct stereoCamParams
{
	//parameters Cam
	float f = 1280.097351; // focal length[pixel]
	float sx = 1.0; // pixel width X [no unit]
	float sy = 1.002822; // pixel width Y [no unit]
	float u0 = 512.788788 - 24; // principal point X [pixel]
	float v0 = 225.899994;// 170.038092 - 40; // principal point Y [pixel] //225.899994 - 40; 
	float b = 0.26; // baseline [meter]
	float cX = 0.0; // camera lateral position [meter]
	float cY = 1.2; // camera height above ground [meter]
	float cZ = 1.7; // camera offset Z [meter]
	float tilt = 0.072; //camera tilt angle [rad] (NOTE: should be adjusted by online tilt angle estimation)
	float fx = f / sx;//focal_length_pixels in X coordinate
	float fy = f / sy;//focal_length_pixels in Y coordinate

					  //parameters Cam
	stereoCamParams()
	{

	}
} stereoParams;

inline void StereoImageBuffer::ImageToWorldCoordinates(StereoImageBuffer::MyObject& obj)
{
	float d = obj.yhighDisp_Index; //consider closest point of the object
	float u, v, vUpline1, vBaseline1;

	if (d > 0)
	{
		if (obj.isLeftLeaning)
			u = obj.xHigh_Index;//if the object is on the left side than consider its rightmost point
		else
			u = obj.xLow_Index;//if the object is on the right side than consider its leftmost point

							   //u = (obj.xLow_Index + obj.xHigh_Index) / 2;

		vUpline1 = obj.yhighDisp_Vmin_Index;//closest point upline //you have to use d = obj.yhighDisp_Index
		vBaseline1 = obj.yhighDisp_Vmax_Index;//closest point baseline //you have to use d = obj.yhighDisp_Index			

		v = vUpline1;
		//compute 3D point
		obj.Zs = (stereoParams.fx * stereoParams.b) / d;
		obj.Xs = (obj.Zs / stereoParams.fx) * (u - stereoParams.u0);
		obj.Ys = (obj.Zs / stereoParams.fy) * (stereoParams.v0 - v); //Upline of the object regarding to the camera position (zero (0) if parelel to the camera)
		float Ys2 = (obj.Zs / stereoParams.fy) * (stereoParams.v0 - vBaseline1); //Upline of the object regarding to the camera position (zero (0) if parelel to the camera)
																				 //float objHeight = obj.Ys - Ys2;

		float Xs2;
		if (obj.isLeftLeaning)
			Xs2 = (obj.Zs / stereoParams.fx) * (obj.xLow_Index - stereoParams.u0);
		else
			Xs2 = (obj.Zs / stereoParams.fx) * (obj.xHigh_Index - stereoParams.u0);

		// correct with camera position and tilt angle
		obj.Xw = obj.Xs + stereoParams.cX; // (-b / 2)
		obj.Zw = obj.Zs *cos(stereoParams.tilt) + obj.Ys *sin(stereoParams.tilt) + stereoParams.cZ;
		obj.Yw = -obj.Zs*sin(stereoParams.tilt) + obj.Ys*cos(stereoParams.tilt) + stereoParams.cY;
		float Yw2 = -obj.Zs*sin(stereoParams.tilt) + Ys2*cos(stereoParams.tilt) + stereoParams.cY;
		float Xw2 = Xs2 + stereoParams.cX; // (-b / 2)

		obj.objHeight = obj.Yw - Yw2;
		obj.objWidth = abs(obj.Xw - Xw2);
		obj.objSize = obj.objHeight * obj.objWidth;
		obj.realDistanceTiltCorrected = sqrt(pow(obj.Xw, 2) + pow(obj.Yw, 2) + pow(obj.Zw, 2));
		obj.directDistance = sqrt(pow(obj.Xs, 2) + pow(obj.Ys, 2) + pow(obj.Zs, 2));
	}
}

#pragma region Stereo Calculations

//% compute 3D point
//Zs = (fx * b) / d;
//Xs = (Zs / fx) * (u - u0);
//Ys = (Zs / fy) * (v0 - v);

//% correct with camera position and tilt angle
//Xw = Xs + cX; % (-b / 2)
//Zw = Zs *cos(tilt) + Ys *sin(tilt) + cZ;
//Yw = -Zs *sin(tilt) + Ys *cos(tilt) + h;

//%reverse sensor(cos(-t) = cos(t), sin(-t) = -sin(t)
//Xs = Xw - cX; % (+b / 2)
//Zs = cos(tilt)*(Zw - cZ) - sin(tilt) * (Yw - h);
//Ys = sin(tilt)*(Zw - cZ) + cos(tilt) * (Yw - h);

//% compute u, v, d
//u = u0 + fx * (Xs / Zs);
//v = v0 -  fy * (Ys / Zs);
//d = (fx * b) / Zs;
inline float StereoImageBuffer::get_v_From_Calibration(float d, float Yw, int& out_v) const
{
	if (d > 0)
	{
		float Zs = (stereoParams.fx * stereoParams.b) / d;
		float Ys = (Yw + Zs*sin(stereoParams.tilt) - stereoParams.cY) / cos(stereoParams.tilt);
		float v = stereoParams.v0 - stereoParams.fy * (Ys / Zs);

		float Zw2 = Zs *cos(stereoParams.tilt) + Ys *sin(stereoParams.tilt) + stereoParams.cZ;

		float out_Yw = -Zs*sin(stereoParams.tilt) + Ys*cos(stereoParams.tilt) + stereoParams.cY;

		out_v = (int)round(v);
		return v;
	}
	else
	{
		out_v = 0;
		return 0;
	}

}

inline float StereoImageBuffer::get_u_From_Calibration(float d, float Xw, int& out_u) const
{
	float Zs = (stereoParams.fx * stereoParams.b) / d;
	float Xs = Xw + stereoParams.cX;
	float u = stereoParams.u0 + stereoParams.fx * (Xs / Zs);

	out_u = (int)round(u);
	return u;
}

inline float StereoImageBuffer::get_d_From_Calibration(float Zw, float Yw, int& out_d) const
{
	float Zs = cos(stereoParams.tilt)*(Zw - stereoParams.cZ) - sin(stereoParams.tilt) * (Yw - stereoParams.cY);
	float d = (stereoParams.fx * stereoParams.b) / Zs;
	out_d = (int)round(d);

	//check and delete
	float Ys = sin(stereoParams.tilt)*(Zw - stereoParams.cZ) + cos(stereoParams.tilt) * (Yw - stereoParams.cY);
	float Zw2 = Zs *cos(stereoParams.tilt) + Ys *sin(stereoParams.tilt) + stereoParams.cZ;
	float Zs2 = (Zw - Ys *sin(stereoParams.tilt) - stereoParams.cZ) / cos(stereoParams.tilt);

	return d;
}

inline float StereoImageBuffer::get_zw_From_Calibration(float d, float Yw) const
{
	if (d > 0)
	{
		float Zs = (stereoParams.fx * stereoParams.b) / d;
		float Ys = (Yw + Zs*sin(stereoParams.tilt) - stereoParams.cY) / cos(stereoParams.tilt);
		float Zw = Zs *cos(stereoParams.tilt) + Ys *sin(stereoParams.tilt) + stereoParams.cZ;

		return Zw;
	}
	else
	{
		return 0;
	}
}

inline void StereoImageBuffer::ImageToWorldCoordinates(float u, float v, float d, float& out_Xw, float& out_Yw, float& out_Zw)
{
	//Image coordinate plane
	float Zs = (stereoParams.fx * stereoParams.b) / d;
	float Xs = (Zs / stereoParams.fx) * (u - stereoParams.u0);
	float Ys = (Zs / stereoParams.fy) * (stereoParams.v0 - v);

	//world coordinate plane
	out_Xw = Xs + stereoParams.cX; // (-b / 2)
	out_Zw = Zs * cos(stereoParams.tilt) + Ys * sin(stereoParams.tilt) + stereoParams.cZ;
	out_Yw = -Zs * sin(stereoParams.tilt) + Ys * cos(stereoParams.tilt) + stereoParams.cY;
}

inline void StereoImageBuffer::WorldToImageCoordinates(float Xw, float Yw, float Zw, float& out_u, float& out_v, float& out_d)
{
	//World to Image coordinate plane
	float Xs = Xw - stereoParams.cX;// % (+b / 2)
	float Zs = cos(stereoParams.tilt)*(Zw - stereoParams.cZ) - sin(stereoParams.tilt) * (Yw - stereoParams.cY);
	float Ys = sin(stereoParams.tilt)*(Zw - stereoParams.cZ) + cos(stereoParams.tilt) * (Yw - stereoParams.cY);

	//Image coordinates to Image
	out_u = stereoParams.u0 + stereoParams.fx * (Xs / Zs);
	out_v = stereoParams.v0 - stereoParams.fy * (Ys / Zs);
	out_d = (stereoParams.fx * stereoParams.b) / Zs;
}

inline void StereoImageBuffer::WorldToImageCoordinates(float Xw, float Yw, float Zw, int& out_u, int& out_v, int& out_d)
{
	//World to Image coordinate plane
	float Xs = Xw - stereoParams.cX;// % (+b / 2)
	float Zs = cos(stereoParams.tilt)*(Zw - stereoParams.cZ) - sin(stereoParams.tilt) * (Yw - stereoParams.cY);
	float Ys = sin(stereoParams.tilt)*(Zw - stereoParams.cZ) + cos(stereoParams.tilt) * (Yw - stereoParams.cY);

	//Image coordinates to Image
	float u = stereoParams.u0 + stereoParams.fx * (Xs / Zs);
	float v = stereoParams.v0 - stereoParams.fy * (Ys / Zs);
	float d = (stereoParams.fx * stereoParams.b) / Zs;

	out_u = (int)round(u);
	out_v = (int)round(v);
	out_d = (int)round(d);
}

inline void StereoImageBuffer::WorldToImagePlane(float Xw, float Yw, float Zw, float& out_Xs, float& out_Ys, float& out_Zs)
{
	//World to Image coordinate plane
	out_Xs = Xw - stereoParams.cX;// % (+b / 2)
	out_Zs = cos(stereoParams.tilt)*(Zw - stereoParams.cZ) - sin(stereoParams.tilt) * (Yw - stereoParams.cY);
	out_Ys = sin(stereoParams.tilt)*(Zw - stereoParams.cZ) + cos(stereoParams.tilt) * (Yw - stereoParams.cY);
}

inline void StereoImageBuffer::ImageToImageCoordinates(float u, float v, float d, float& out_Xs, float& out_Ys, float& out_Zs)
{
	//Image to Image coordinate plane
	out_Zs = (stereoParams.fx * stereoParams.b) / d;
	out_Zs = (out_Zs / stereoParams.fx) * (u - stereoParams.u0);
	out_Ys = (out_Zs / stereoParams.fy) * (stereoParams.v0 - v);
}

#pragma endregion Stereo Calculations

inline int StereoImageBuffer::getNumObjectsDeleted(vector<MyObject, NAlloc<MyObject>>& objects) const
{
	int deletedObjNum = 0;
	for (int i = 0; i < Global_currentNumOfObjects; i++)
	{
		if (objects[i].isDeleted) deletedObjNum++;
	}

	return deletedObjNum;
}

inline void StereoImageBuffer::printObject(MyObject &obj) const
{
	cout << endl << "LabelNum:" << obj.labelNum << " MergedLNum:" << obj.mergedLabelNum << "(" << obj.isMerged << ")" << " InclinedLNum:" << obj.mergedLabelNumInclined << "(" << obj.isMergedInclined << ") Left:" << obj.isLeftSide << endl;
	cout << "xLow:" << obj.xLow_Index << " xHigh:" << obj.xHigh_Index << " ylowDisp:" << obj.ylowDisp_Index << " yhighDisp:" << obj.yhighDisp_Index << " reverse:" << obj.isReverseSide << " xlowlow:" << obj.xLow_Index_lowDisp << " xlowhi:" << obj.xLow_Index_highDisp << endl;
	cout << "ylowDisp_Vmin:" << obj.ylowDisp_Vmin_Index << " yhighDisp_Vmin:" << obj.yhighDisp_Vmin_Index << " ylowDisp_Vmax:" << obj.ylowDisp_Vmax_Index << " yhighDisp_Vmax:" << obj.yhighDisp_Vmax_Index << endl;
	cout << fixed << std::setprecision(2) << "(udisp) m:" << obj.udisp_slope << " b:" << obj.udisp_slope_b << " rad:" << obj.udisp_radian << " deg:" << obj.udisp_degree << endl;
	cout << fixed << std::setprecision(2) << "Xw:" << obj.Xw << " Yw(up):" << obj.Yw << " Zw:" << obj.Zw << " dDist:" << obj.directDistance << " rDist:" << obj.realDistanceTiltCorrected << " size:" << obj.objSize << endl;
	cout << fixed << std::setprecision(2) << "Xw2:" << abs(obj.objWidth - obj.Xw) << " Yw(base):" << obj.Yw - obj.objHeight << " Xs:" << obj.Xs << " Ys(up):" << obj.Ys << " Zs:" << obj.Zs << endl;

	//cout << "RoadLine_m:" << Global_vDisparityGlobals->bestRoadLine_m << " RoadLine_b:" << Global_vDisparityGlobals->bestRoadLine_b << " vIntersect:" << Global_vDisparityGlobals->bestRoadLine_vIntersection << " dInterSect:" << Global_vDisparityGlobals->bestRoadLine_dInterSection << endl << endl;
}

void StereoImageBuffer::cvPolyfit(cv::Mat &src_x, cv::Mat &src_y, cv::Mat &dst, int order) const
{
	//Polynomial regression beta = inv((XT*X))*XT*y
	CV_FUNCNAME("cvPolyfit");
	__CV_BEGIN__;
	{
		CV_ASSERT((src_x.rows>0) && (src_y.rows>0) && (src_x.cols == 1) && (src_y.cols == 1)
			&& (dst.cols == 1) && (dst.rows == (order + 1)) && (order >= 1));
		Mat X;
		X = Mat::zeros(src_x.rows, order + 1, CV_32FC1);
		Mat copy;
		for (int i = 0; i <= order; i++)
		{
			copy = src_x.clone();
			pow(copy, i, copy);
			Mat M1 = X.col(i);
			copy.col(0).copyTo(M1);
		}
		Mat X_t, X_inv;
		transpose(X, X_t);
		Mat temp = X_t*X;
		Mat temp2;
		invert(temp, temp2);
		Mat temp3 = temp2*X_t;
		Mat W = temp3*src_y;

		//cout << "PRINTING INPUT AND OUTPUT FOR VALIDATION AGAINST MATLAB RESULTS\n";
		//cout << "SRC_X: " << src_x << endl;
		//cout << "SRC_Y: " << src_y << endl;
		//cout << "X: " << X << endl;
		//cout << "X_T: " << X_t << endl;
		//cout << "W:" << W << endl;
		dst = W.clone();
	}
	__CV_END__;
}

void StereoImageBuffer::testPolyFit() const
{

	int start_s = clock(); // clock start

	int rows = 10;
	int cols = 1;

	Mat time = Mat_<float>(rows, cols, CV_64F);
	Mat x = Mat_<float>(rows, cols, CV_64F);
	Mat y = Mat_<float>(rows, cols, CV_64F);
	Mat z = Mat_<float>(rows, cols, CV_64F);

	int count = 0;

	while (count<25) // to fill the data matrices and compute simultaneoulsy
	{
		count++;
		// data filled here
		time.push_back(count*1.0f);
		x.push_back(count*1.0f);
		y.push_back(count*count*1.0f);
		z.push_back(count*1.0f);


		if (time.rows>4) // if the data matrices have more than 4 elements then I remove old elements
		{
			// data matrices resized here
			Mat temp1 = time.colRange(0, time.cols).rowRange(1, time.rows).clone();
			Mat temp2 = x.colRange(0, x.cols).rowRange(1, x.rows).clone();
			Mat temp3 = y.colRange(0, y.cols).rowRange(1, y.rows).clone();
			Mat temp4 = z.colRange(0, z.cols).rowRange(1, z.rows).clone();

			time = temp1;
			x = temp2;
			y = temp3;
			z = temp4;
		}

		//cout << "X: " << time.rows << "[ ";
		//for (int i = 0; i < time.rows; i++)
		//{
		//	cout << time.at<float>(i, 0) << " ";
		//}
		//cout << "]" << endl;

		//cout << "Y1: " << time.rows << "[ ";
		//for (int i = 0; i < time.rows; i++)
		//{
		//	cout << x.at<float>(i, 0) << " ";
		//}
		//cout << "]" << endl;

		//cout << "Y2: " << time.rows << "[ ";
		//for (int i = 0; i < time.rows; i++)
		//{
		//	cout << y.at<float>(i, 0) << " ";
		//}
		//cout << "]" << endl;

		//cout << "Y3: " << time.rows << "[ ";
		//for (int i = 0; i < time.rows; i++)
		//{
		//	cout << z.at<float>(i, 0) << " ";
		//}
		//cout << "]" << endl;

		// order of data matrix
		int fit_orderX = 1;
		int fit_orderY = 2;
		int fit_orderZ = 1;

		// matrices to store computed coefficients
		Mat fit_weightsX(fit_orderX + 1, 1, CV_32FC1);
		Mat fit_weightsY(fit_orderY + 1, 1, CV_32FC1);
		Mat fit_weightsZ(fit_orderZ + 1, 1, CV_32FC1);

		// polyFit function used to get coeffecients
		cvPolyfit(time, x, fit_weightsX, fit_orderX);
		cvPolyfit(time, y, fit_weightsY, fit_orderY);
		cvPolyfit(time, z, fit_weightsZ, fit_orderZ);


		//cout << roundf(fit_weightsX.at<float>(1, 0) * 1000) / 1000 << "x + " << roundf(fit_weightsX.at<float>(0, 0) * 1000) / 1000 << endl;
		//cout << roundf(fit_weightsZ.at<float>(1, 0) * 1000) / 1000 << "x + " << roundf(fit_weightsZ.at<float>(0, 0) * 1000) / 1000 << endl;
		//cout << roundf(fit_weightsY.at<float>(1, 0) * 1000) / 1000 << "x + " << roundf(fit_weightsY.at<float>(2, 0) * 1000) / 1000 << "x2 +" << roundf(fit_weightsY.at<float>(0, 0) * 1000) / 1000 << endl;
		//cout << "-----------" << endl;

		//// computed data printed here
		//cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
		//cout << "Weights of fit of order of X-data are: " << fit_weightsX << endl;
		//// X = Ax + Bx * t ———–> This one is of first order (so 2 coefficients)
		//cout << "Ax is –> " << roundf(fit_weightsX.at<float>(0,0) * 1000) / 1000 << endl; // rounded coefficients
		//cout << "Bx is –> " << roundf(fit_weightsX.at<float>(1,0) * 1000) / 1000 << endl;
		//cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
		//cout << "Weights of fit of order of Y-data are: " << fit_weightsY << endl;
		//// Y = Ay + By * t + Cy * t^2 ———–> This one is of second order (so 3 coefficients)
		//cout << "Ay is –> " << roundf(fit_weightsY.at<float>(0,0) * 1000) / 1000 << endl; // rounded coefficients
		//cout << "By is –> " << roundf(fit_weightsY.at<float>(1,0) * 1000) / 1000 << endl;
		//cout << "Cy is –> " << roundf(fit_weightsY.at<float>(2,0) * 1000) / 1000 << endl;
		//cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
		//cout << "Weights of fit of order of Z-data are: " << fit_weightsZ << endl;
		//// Z = Az + Bz * t ———–> This one is of first order (so 2 coefficients)
		//cout << "Az is –> " << roundf(fit_weightsZ.at<float>(0,0) * 1000) / 1000 << endl; // rounded coefficients
		//cout << "Bz is –> " << roundf(fit_weightsZ.at<float>(1,0) * 1000) / 1000 << endl;
		//cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
		// print end

	}// while end

	 // clock stopped to get execution time
	int stop_s = clock();
	// execution time in milliseconds
	//cout << "time taken to execute this program is : " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << " millisecond" << endl;
}

#ifdef WITH_OCCUPANCY_GRID
inline void StereoImageBuffer::ImageToWorldCoordinates(StereoImageBuffer::uOccupancyCell& obj)
{

	//float v = stereoParams.v0;//Means that Ys = 0 (we only consider the direct distance (perpendicular) with respect to camera position, which will be used to determine Zw tilt corrected
	//float Ys = 0;//since (stereoParams.v0 - v) = 0

	float upDepth = (stereoParams.fx * stereoParams.b) / obj.upD;//= Zs
	float downDepth = (stereoParams.fx * stereoParams.b) / obj.downD;
	//important !!! should we use "Ys = (Zs / stereoParams.fy) * (stereoParams.v0 - v);" as well ??? since Zw = Zs *cos(stereoParams.tilt) + Ys *sin(stereoParams.tilt) + stereoParams.cZ; ???
	//float surfaceLeftUpYs = (upDepth / stereoParams.fy) * (stereoParams.v0 - v); ???

	obj.surfaceLeftUpZ = upDepth*cos(stereoParams.tilt) + stereoParams.cZ;
	obj.surfaceLeftUpX = (upDepth / stereoParams.fx) * (obj.leftU - stereoParams.u0) + stereoParams.cX;
	//obj.surfaceLeftUpZ = sqrt(pow(obj.surfaceLeftUpZ, 2) + pow(obj.surfaceLeftUpX, 2));//important!!! distance should be calculated according to x and z position in order to keep position when yaw angle changes

	obj.surfaceLeftDownZ = downDepth*cos(stereoParams.tilt) + stereoParams.cZ;
	obj.surfaceLeftDownX = (downDepth / stereoParams.fx) * (obj.leftU - stereoParams.u0) + stereoParams.cX;
	//obj.surfaceLeftDownZ = sqrt(pow(obj.surfaceLeftDownZ, 2) + pow(obj.surfaceLeftDownX, 2));

	obj.surfaceRightUpZ = upDepth*cos(stereoParams.tilt) + stereoParams.cZ;	//obj.surfaceLeftUpZ = obj.surfaceRightUpZ
	obj.surfaceRightUpX = (upDepth / stereoParams.fx) * (obj.rightU - stereoParams.u0) + stereoParams.cX;
	//obj.surfaceRightUpZ = sqrt(pow(obj.surfaceLeftUpZ, 2) + pow(obj.surfaceRightUpX, 2));

	obj.surfaceRightDownZ = downDepth*cos(stereoParams.tilt) + stereoParams.cZ;	//obj.surfaceLeftDownZ = obj.surfaceRightDownZ
	obj.surfaceRightDownX = (downDepth / stereoParams.fx) * (obj.rightU - stereoParams.u0) + stereoParams.cX;
	//obj.surfaceRightDownZ = sqrt(pow(obj.surfaceLeftDownZ, 2) + pow(obj.surfaceRightDownX, 2));
}

inline float StereoImageBuffer::my_round(float x, unsigned int digits, bool recursive)
{
	if (!recursive)
	{
		float fac = pow(10, digits);
		return round(x*fac) / fac;
	}
	else
	{
		if (digits > 0) {
			return my_round(x*10.0, digits - 1, true) / 10.0;
		}
		else {
			return round(x);
		}
	}
}
// ReSharper disable CppMemberFunctionMayBeStatic
inline void StereoImageBuffer::set_occupancyUDisparityMap() const
{
	/*
	#pragma region trivia
	cv::Mat occupanyMap_Z = cv::Mat::zeros(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, Global_Current_DenseDisparity_Image.type());
	cv::Mat occupanyMap_Y = cv::Mat::zeros(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_32FC3);
	cv::Mat occupanyMap_Y_FC1 = cv::Mat::zeros(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_32FC1);
	cv::Mat occupanyMap_X = cv::Mat::zeros(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_32FC3);
	cv::Mat occupanyMap_R = cv::Mat::zeros(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, Global_Current_DenseDisparity_Image.type());

	cv::Mat uDisp_Obstacle = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, Global_uDisparityGlobals->uDisparityImage.type());
	cv::Mat uDisp_Road = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, Global_uDisparityGlobals->uDisparityImage.type());

	cv::Mat originalLeftIMG = Global_Current_Left_Image;
	cv::Mat originalRightIMG = Global_Current_Right_Image;

	cv::Mat dDisparity = Global_Current_DenseDisparity_Image;
	cv::Mat vDisparity = Global_vDisparityGlobals->vDisparityImage;
	cv::Mat uDisparity = Global_uDisparityGlobals->uDisparityImage;


	const unsigned char* row_disparity;
	unsigned char* cell_occDisparity_Z;
	float* cell_occDisparity_Y;
	float* cell_occDisparity_Y_FC1;
	float* cell_occDisparity_X;
	unsigned char* cell_occDisparity_R;

	for (int v = 0; v< Global_dDisparityParams->totalRows; v++)
	{
	row_disparity = Global_Current_DenseDisparity_Image.ptr<const unsigned char>(v);

	for (int u = 0; u < Global_dDisparityParams->totalCols; u++)
	{
	int d = row_disparity[u];
	cell_occDisparity_Z = occupanyMap_Z.ptr<unsigned char>(v, u);

	cell_occDisparity_Y = occupanyMap_Y.ptr<float>(v, u);
	cell_occDisparity_Y_FC1 = occupanyMap_Y_FC1.ptr<float>(v, u);
	cell_occDisparity_X = occupanyMap_X.ptr<float>(v, u);
	cell_occDisparity_R= occupanyMap_R.ptr<unsigned char>(v, u);

	if (d == 0)
	{
	cell_occDisparity_Z[0] = 60;
	cell_occDisparity_X[0] = 60;
	cell_occDisparity_Y[0] = 0.33;
	cell_occDisparity_Y_FC1[0] = 0.33;
	cell_occDisparity_R[0] = 60;
	}
	else
	{
	float Zs_f = (fx * stereoParams->b) / d;
	float Xs_f = (Zs_f / fx) * (u - stereoParams->u0);
	float Ys_f = (Zs_f / fy) * (stereoParams->v0 - v);

	int Zs = round(Zs_f);
	int Xs = round(Xs_f);
	int Ys = round(Ys_f);

	float Xw_f = Xs_f + stereoParams->cX;
	float Zw_f = Zs_f *cos(stereoParams->tilt) + Ys_f *sin(stereoParams->tilt) + stereoParams->cZ;
	float Yw_f = -Zs_f*sin(stereoParams->tilt) + Ys_f*cos(stereoParams->tilt) + stereoParams->cY;
	//float Yw_f = Ys_f + stereoParams->cY;

	int Zw = round(Zw_f);
	int Xw = round(Xw_f);
	int Yw = round(Yw_f);

	float real_dist = sqrt(pow(Xw_f, 2) + pow(Yw_f, 2) + pow(Zw_f, 2));
	int dist = round(real_dist);

	cell_occDisparity_Z[0] = Zw_f;
	cell_occDisparity_X[0] = Xw_f;
	cell_occDisparity_Y[0] = Yw_f;
	cell_occDisparity_Y_FC1[0] = Yw_f;
	if (Yw < 4)
	{
	if (Yw < -1)
	{
	cell_occDisparity_Y_FC1[0] = -1;
	cell_occDisparity_Y[0] = -1;
	cell_occDisparity_Y[1] = 0;
	cell_occDisparity_Y[2] = pow(Yw, 4);
	}
	else
	{
	cell_occDisparity_Y[1] = 0;
	cell_occDisparity_Y[2] = pow(Yw, 4);
	}

	cell_occDisparity_X[1] = 0;
	cell_occDisparity_X[2] = pow(Yw,4);
	}
	else
	{
	if (Yw < 15)
	{
	cell_occDisparity_Y[1] = pow(Yw, 2);
	cell_occDisparity_Y[2] = 0;
	cell_occDisparity_X[1] = pow(Yw, 2);
	cell_occDisparity_X[2] = 0;
	}
	else
	{
	cell_occDisparity_Y[1] = 0;
	cell_occDisparity_Y[2] = 0;
	cell_occDisparity_X[1] = 0;
	cell_occDisparity_X[2] = 0;
	}
	}
	cell_occDisparity_R[0] = real_dist;
	}
	}
	}

	Mat occupanyMap_Y_UC;
	double minValY, maxValY;
	cv::minMaxLoc(occupanyMap_Y_FC1, &minValY, &maxValY);
	occupanyMap_Y_FC1.convertTo(occupanyMap_Y_UC, CV_8UC1, 255 / (maxValY - minValY));
	Mat occupanyMap_Y_FC1_ColorMap;
	applyColorMap(occupanyMap_Y_UC, occupanyMap_Y_FC1_ColorMap, COLORMAP_HOT);
	#pragma endregion trivia
	*/

/*
#pragma region trivia2
	Mat imgZs = cv::Mat::zeros(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_32FC1);
	Mat imgZs_UC = cv::Mat::zeros(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_32FC1);
	Mat imgXs = cv::Mat::zeros(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_32FC1);
	Mat imgYs = cv::Mat::zeros(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_32FC1);
	Mat imgZw = cv::Mat::zeros(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_32FC1);
	Mat imgZw_XZ = cv::Mat::zeros(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_32FC1);
	Mat imgZw_XYZ = cv::Mat::zeros(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_32FC1);
	Mat imgZw_checkOccupancy = cv::Mat::zeros(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_32FC1);

	Mat imgXw = cv::Mat::zeros(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_32FC1);
	Mat imgYw = cv::Mat::zeros(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_32FC1);
	Mat imgYw2 = cv::Mat::zeros(Global_dDisparityParams->totalRows, Global_dDisparityParams->totalCols, CV_8UC1);
	Mat imgDense = Global_Current_DenseDisparity_Image;
	Mat imgDenseF = Global_Current_DenseDisparity_Image32F;
	Mat imgLeft = Global_Current_Left_Image;
	Mat imgRight = Global_Current_Right_Image;

	const float* checkRow_disparity32F;
	const unsigned char* checkRow_disparityUC;
	for (int v = 0; v< Global_dDisparityParams->totalRows; v++)
	{
		checkRow_disparity32F = Global_Current_DenseDisparity_Image32F.ptr<const float>(v);
		checkRow_disparityUC = Global_Current_DenseDisparity_Image.ptr<const unsigned char>(v);

		for (int u = 0; u < Global_dDisparityParams->totalCols; u++)
		{
			float d = checkRow_disparity32F[u];
			float di = (float)checkRow_disparityUC[u];

			if (di <= 0) continue;

			float Zs = (stereoParams.fx * stereoParams.b) / d;
			float Xs = (Zs / stereoParams.fx) * (u - stereoParams.u0);
			float Ys = (Zs / stereoParams.fy) * (stereoParams.v0 - v); //Upline of the object regarding to the camera position (zero (0) if parelel to the camera)

																	   // correct with camera position and tilt angle
			float Xw = Xs + stereoParams.cX; // (-b / 2)
			float Zw = Zs *cos(stereoParams.tilt) + Ys *sin(stereoParams.tilt) + stereoParams.cZ;
			float Yw = -Zs*sin(stereoParams.tilt) + Ys*cos(stereoParams.tilt) + stereoParams.cY;

			imgZs.at<float>(v, u) = Zs;
			imgXs.at<float>(v, u) = Xs;
			imgYs.at<float>(v, u) = Ys;

			imgZw.at<float>(v, u) = Zw;
			imgXw.at<float>(v, u) = Xw;
			imgYw.at<float>(v, u) = Yw;

			float Zs_UC = (stereoParams.fx * stereoParams.b) / di;
			imgZs_UC.at<float>(v, u) = Zs_UC;

			float XZ_distance = sqrt(pow(Zw, 2) + pow(Xw, 2));
			imgZw_XZ.at<float>(v, u) = XZ_distance;

			float XYZ_distance = sqrt(pow(Zw, 2) + pow(Xw, 2) + pow(Yw, 2));
			imgZw_XYZ.at<float>(v, u) = XYZ_distance;

			int Yw2;

			if (Yw < 0)
				Yw2 = 0;
			else
				Yw2 = (int)(Yw);

			imgYw2.at<char>(v, u) = Yw2 * 10;

			//check occupancy ImagetoWordCoordinate values
			//obj.surfaceLeftUpZ = upDepth*cos(stereoParams.tilt) + stereoParams.cZ;
			//obj.surfaceLeftUpX = (upDepth / stereoParams.fx) * (obj.leftU - stereoParams.u0) + stereoParams.cX;
			//obj.surfaceLeftUpZ = sqrt(pow(obj.surfaceLeftUpZ, 2) + pow(obj.surfaceLeftUpX, 2));//important!!! distance should be calculated according to x and z position in order to keep position when yaw angle changes
			float Zw_Occupancy = Zs *cos(stereoParams.tilt) + stereoParams.cZ;
			float Zw_Occupancy_xs = sqrt(pow(Zw_Occupancy, 2) + pow(Xw, 2));
			imgZw_checkOccupancy.at<float>(v, u) = XYZ_distance;

		}
	}

	float Zw = 15.4;
	float Yw = 2.57;
	float Xw = 2.0;
	int out_dint, out_vint, out_uint;
	float out_Xw, out_Yw, out_Zw;

	float dx = get_d_From_Calibration(Zw, Yw, out_dint);
	float vx = get_v_From_Calibration(dx, Yw, out_vint);
	ImageToWorldCoordinates(0.0, vx, dx, out_Xw, out_Yw, out_Zw);

#pragma endregion trivia2

*/
	const unsigned char* row_disparity;
	const float* row_disparity32F;

	Global_occupancyGridGlobals->occupancyGridMap.set_OccupancyGridMatrix(*Global_occupancyGridParams);

#ifdef WITH_SLAM
	int slamIndex = getCurrentImageNumInt();
	int imgIndex = (slamIndex - 19) / 10;

	Global_occupancyGridParams->yawDegree = manualAngles[imgIndex];
	Global_occupancyGridParams->xVelocity = manualX[imgIndex];
	Global_occupancyGridParams->zVelocity = manualZ[imgIndex];

	//Global_occupancyGridParams->yawDegree = manualAngles[Global_currentIMGIndex];
	//Global_occupancyGridParams->xVelocity = manualX[Global_currentIMGIndex];
	//Global_occupancyGridParams->zVelocity = manualZ[Global_currentIMGIndex];

	Global_occupancyGridParams->setRotationMatrix();
#endif

	cv::Mat occupanyMapU_UC1 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, Global_uDisparityGlobals->uDisparityImage.type());
	cv::Mat occupanyMapU_FC1 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, CV_32FC1);
	cv::Mat occupanyMapU_Visibility_FC1 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, CV_32FC1);
	cv::Mat occupanyMapU_Confidence_FC1 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, CV_32FC1);
	cv::Mat occupanyMapU_PDF_FC1 = cv::Mat::zeros(Global_uDisparityParams->totalRows, Global_uDisparityParams->totalCols, CV_32FC1);

	Mat check_occupancy, check_occupancy_t;

	cv::Mat uDisparityMap = Global_uDisparityGlobals->uDisparityImage;

	const unsigned char* row_udisparity;


	unsigned char* cell_uDisparity;
	//search between defined heights
	float maxYw = 1.5;//meter
	float minYw = 0.0;//meter
					  //search between defined heights

	float RoUD;//Estimation of the Confidence of Observation
	float PVu;//The probibility of visibility
	float PCu;//probability density function of Confidence
	float POu;//final probability density function with visibility
	float POu_posterior;//final probability density function with visibility using cell history probility, joint probility

	float *cellU_Occupancy;
	float *cellU_Visibility;
	float *cellU_Confidence;
	float *cellU_Occupancy_PDF;
	float *cellU_Occupancy_PDF_Thresholded;

	unsigned char* cell_uDisp_Obstacle;
	unsigned char* cell_uDisp_Road;

	uOccupancyCell* myuOccupancyCell;

	int cellIndex = -1;
	int cellMapIndex = -1;//used on the 360 degree SLAM map
						  //int startD = round((stereoParams.fx * stereoParams.b) / this->Global_occupancyGridParams->ZWorldLimit); //where Zw = 40m(Zworldlimit)
	int startD; //where Zw = 40m(Zworldlimit)
	get_d_From_Calibration(this->Global_occupancyGridParams->ZWorldLimit, stereoParams.cY, startD);

	int u, d;

	Mat check_disp = Global_Current_DenseDisparity_Image;
	Mat check_disp2 = Global_Current_DenseDisparity_Image32F;
	Mat check_left = Global_Current_Left_Image;

	for (d = startD; d< Global_uDisparityParams->totalRows; d++)
	{
		float Zs = (stereoParams.fx * stereoParams.b) / d;//will be used to find v coordinate of the base and the maxH
		float Ys_h = (maxYw + Zs*sin(stereoParams.tilt) - stereoParams.cY) / cos(stereoParams.tilt);
		float Ys_v0 = (minYw + Zs*sin(stereoParams.tilt) - stereoParams.cY) / cos(stereoParams.tilt);

		//int NpUD = round((fx / fy)*((Ys*d) / stereoParams.b));//num of possible pixels
		int vDown = (int)round(stereoParams.v0 - stereoParams.fy * (Ys_v0 / Zs));
		if (vDown >= Global_dDisparityParams->totalRows)
			vDown = Global_dDisparityParams->totalRows;

		//int NpUD_perrolaz = round(fy * (Ys / Zs));//(Vo-Vh)//Vh=Ys //num of possible pixels
		int NpUD_perrolaz = round(stereoParams.fy * ((Ys_h - Ys_v0) / Zs));//(Vo-Vh)//Vh=Ys //num of possible pixels

		float Zw = Zs *cos(stereoParams.tilt) + Ys_h *sin(stereoParams.tilt) + stereoParams.cZ;
		//float Ysh = maxYw;

		int vUp = (int)round(stereoParams.v0 - stereoParams.fy * (Ys_h / Zs));
		if (vUp < 0)
		{
			vUp = 0;
		}

		int vBase = get_v_From_VDisparity_RoadProfile(d); //round(d*Global_vDisparityGlobals->bestRoadLine_m + Global_vDisparityGlobals->bestRoadLine_b); //Global_dDisparityParams->totalRows - 1;// stereoParams->v0;//I think we can use the object base found by vDisparity

		int oldNpUD = vBase - vUp;
		int NpUD = vDown - vUp;
		int NpUD_perrolaz2 = vDown - vUp;

		//cout << "d:" << d << " (Zs:" << Zs << " - Zw:" << Zw << ") maxY:" << maxYw << " NpUD:" << oldNpUD << " (" << vUp << "-" << vBase << ") - NpUD_p:" << NpUD_perrolaz << "-" << NpUD_perrolaz2 << " (" << vUp << "-" << vDown << ") v0:" << stereoParams.v0 << endl;

		//int totalCols_Resolution = (int)round(Global_uDisparityParams->totalCols / (float)Global_occupancyGridParams->uDisp_uResolution);


		for (u = 15; u < Global_uDisparityParams->totalCols; u += Global_occupancyGridParams->uDisp_uResolution)
		{
			//cout << "Error at u:" << u << endl;
			//if(u == 1051)
			//	cout << "Error at u:" << u << endl;
			//save udisp occupancy cells to array
			//int value = row_udisparity[u];

			cellIndex = cellIndex + 1;

			float NvUD = 0;//num of visible pixels
			float NoUD = 0;//num of observed pixels

			float real_dmin = Global_numDisparities;
			float real_dmax = 0;

			for (int v = vUp; v < vDown; v++)
			{
				//if (u == 1051 && v == 543)
				//	cout << "Error at v:" << v << endl;

				row_disparity = Global_Current_DenseDisparity_Image.ptr<const unsigned char>(v);
				row_disparity32F = Global_Current_DenseDisparity_Image32F.ptr<const float>(v);//important!! we want precise disparities

				int IDuv = row_disparity[u]; //if IDuv > d then it is occluded, if IDuv == 0 then it is not visible, if IDuv == d then it is observed - Perrollaz 2010-2012
				float IDuvf = row_disparity32F[u];

				if (IDuvf < 0)
					IDuvf = IDuv;

				if (IDuv > 0 && IDuv <= d)
				{
					NvUD++;

					if (IDuv == d)
					{
						if (IDuvf < real_dmin && IDuvf > 0)
							real_dmin = IDuvf;

						if (IDuvf > real_dmax)
							real_dmax = IDuvf;

						NoUD++;
					}
				}
			}

			if (NoUD == 0)
			{
				real_dmin = (float)d;
				real_dmax = (float)d;
			}

			if (NpUD > 0)
				PVu = NvUD / NpUD;
			else
				PVu = 0;

			if (NvUD > 0)
				RoUD = NoUD / NvUD;
			else
				RoUD = 0;

			float uLeft = u - Global_occupancyGridParams->uDisp_uResolution + 1;
			float uRight = u;

			float dUp, dDown;
			//dUp = real_dmin - 1.0;
			dDown = real_dmax;

			//use uniform distance
			//dUp = real_dmin - (1 - (real_dmax - real_dmin)); --> we are going to use uniform distance 

			if (PVu > 0)
			{
				float dDown_Zw = get_zw_From_Calibration(dDown, stereoParams.cY);
				float uniformDistance_Zw = 0.85;
				float dUp_Zw = dDown_Zw + uniformDistance_Zw;

				int dUp_int;
				dUp = get_d_From_Calibration(dUp_Zw, stereoParams.cY, dUp_int);
				//}
				//else
				//{
				//	dUp = real_dmin - (1 - (real_dmax - real_dmin));
				//}

				//dUp = real_dmin - (1 - (real_dmax - real_dmin));
				if (dUp <= 0)
					dUp = 0.1;
				//use uniform distance
				PCu = 1 - exp(-(RoUD / Global_occupancyGridParams->uDisp_tO));//Confidence should quickly grow with respect to the number of observed pixels

																			  //PVu*PCu*(1 - FP) = P(Ou|Vu=v,Cu=c) --> if a cell is visible then full confidence that an obstacle is observed and the only way that a cell is not occupied is the false positives
																			  //PVu*(1 - PCu)*FN = P(Ou|Vu=v,Cu=0) --> a cell can only be occupied, when nothing is observed, if there was a false negative
																			  //(1 - PVu)*0.5    = P(Ou|Vu=0,Cu=c) --> if the cell is not visible, nothing is known about its occupancy (%50 means chance, not known)

																			  //create probablity density function where,
																			  //P(OU) = E(v,c)( P(VU =v) * P(CU =c) * P(OU|VU =v,CU =c) )
																			  //P(OU) = P(Vu) * P(Cu) * (1 − PFP) + P(Vu) * (1 − P(Cu)) * PFN + (1 − P(Vu)) * 0.5
																			  //P(OU) = P(VU = 1) * P(CU = 1) * (1 − PFP) + P(VU = 1) * (1 − P(CU = 1)) * PFN + (1 − P(VU = 1)) * 0.5

																			  //POu = PVu*PCu*(1 - Global_occupancyGridParams->uDisp_probabilityFP) + PVu*(1 - PCu)*Global_occupancyGridParams->uDisp_probabilityFN + (1 - PVu)*Global_occupancyGridParams->AprioriCellOccupancy;
				myuOccupancyCell = new uOccupancyCell(PVu, PCu, d, u, uLeft, uRight, dUp, dDown, cellIndex);

				myuOccupancyCell->setGridPoints(*Global_occupancyGridParams);


				if (myuOccupancyCell->isValid)
				{
					Global_occupancyGridGlobals->occupancyGridMap.world2grid(*myuOccupancyCell, *Global_occupancyGridParams);

					//check_occupancy = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix;
#ifdef WITH_SLAM
					//if (Global_occupancyGridParams->useJointProbilities)//delete later, just for testing
					//{
					//	Mat check_PCu, check_PVu;
					//	check_PCu = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixPCu;
					//	check_PVu = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixPVu;
					//	//cout << "";
					//}

					//if (!(Global_occupancyGridParams->useBilinearInterpolation))//delete later, just for testing
					//	check_occupancy_t = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrixSLAM;
					////set_OccupancyGridSLAMMatrix_ByBilinearInterpolation

#else
					check_occupancy_t = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix_Thresholded;
#endif
					//cout << "";
				}
				//else
				//{
				//	cout << myuOccupancyCell->index << " is not valid" << endl;
				//}
			}

		}

		//cout <<  "";
	}

	//cout << "";
	//double minVal, maxVal;
	//cv::minMaxLoc(occupanyMapU_FC1, &minVal, &maxVal);
	//occupanyMapU_FC1.convertTo(occupanyMapU_UC1, CV_8UC1, 255 / (maxVal - minVal));
	//Mat occupanyMap_UC_ColorMap;
	//applyColorMap(occupanyMapU_UC1, occupanyMap_UC_ColorMap, COLORMAP_HOT);

	//myuOccupancyCell = Global_occupancyGridGlobals->uDispOccupancyCells[1500].get();
	//int numOfCells = Global_occupancyGridGlobals->uDispOccupancyCells.size();

	//Global_occupancyGridGlobals->occupancyGridMap.world2gridIndex(*myuOccupancyCell, *Global_occupancyGridParams);
	//Global_occupancyGridGlobals->occupancyGridMap.world2grid(*myuOccupancyCell, *Global_occupancyGridParams);

	//float rounded = my_round(0.559, 1, true);
	//Mat test_occupancyMatrix = Global_occupancyGridGlobals->occupancyGridMap.occupancyMatrix;
	//std::cout << "ok... " << rounded << endl;
}
inline void StereoImageBuffer::set_occupancyDDisparityMap() const
{

}
#endif

#ifdef WITH_DOCK_DETECTION
inline void StereoImageBuffer::getMiddleRadian(MyLine& L1, MyLine& L2, float &difRadian, float &middleRadian) const
{
	if (L2.signMinus != L1.signMinus) //equilize sign of degrees  (exclude if (L2.signMinus == L1.signMinus))
	{
		if (L1.radian < 0)
		{
			difRadian = (L1.radian - L2.radian + CV_PI) / 2;// (90 - L2.radian) + (90 + L1.radian)
			middleRadian = L2.radian + difRadian;

			if (middleRadian >(CV_PI / 2))
			{
				middleRadian = L1.radian - middleRadian + (CV_PI / 2);
			}
		}
		else
		{
			difRadian = (L2.radian - L1.radian + CV_PI) / 2;// (90 - L1.radian) + (90 + L2.radian)
			middleRadian = L1.radian + difRadian;

			if (middleRadian > (CV_PI / 2))
			{
				middleRadian = L2.radian - middleRadian + (CV_PI / 2);
			}
		}
	}
	else
	{
		difRadian = fabs(L2.radian - L1.radian);
		middleRadian = (L2.radian + L1.radian) / 2;
	}
}

inline void StereoImageBuffer::getAngleBetweenlines(MyLine& L1, MyLine& L2, float &difRadian, float &difDegree) const
{
	if (L2.signMinus != L1.signMinus) //equilize sign of degrees  (exclude if (L2.signMinus == L1.signMinus))
	{
		if (L1.radian < 0)
		{
			difRadian = (L1.radian - L2.radian + CV_PI) / 2;// (90 - L2.radian) + (90 + L1.radian)
			difDegree = difRadian*(180 / CV_PI);
		}
		else
		{
			difRadian = (L2.radian - L1.radian + CV_PI) / 2;// (90 - L1.radian) + (90 + L2.radian)
			difDegree = difRadian*(180 / CV_PI);
		}
	}
	else
	{
		difRadian = fabs(L2.radian - L1.radian);
		difDegree = difRadian*(180 / CV_PI);
	}
}

inline StereoImageBuffer::MyLine StereoImageBuffer::merge2Lines(MyLine& L1, MyLine& L2) const
{
	Mat temp1, temp2, temp11, temp22;
	temp1.create(cv::Size(64, 960), CV_8UC3);
	temp2.create(cv::Size(64, 960), CV_8UC3);
	temp11.create(cv::Size(64, 960), CV_8UC3);
	temp22.create(cv::Size(64, 960), CV_8UC3);
	drawLine(L1, temp1);
	drawLine(L2, temp1);

	MyLine L3;


	float middleRadian, difRadian, difDegree;
	//if (L2.signMinus != L1.signMinus) //equilize sign of degrees  (exclude if (L2.signMinus == L1.signMinus))
	//{

	//	if (L1.radian < 0)
	//	{
	//		float dif = (L1.radian - L2.radian + CV_PI) / 2;// (90 - L2.radian) + (90 + L1.radian)
	//		middleRadian = L2.radian + dif;

	//		if (middleRadian > (CV_PI / 2))
	//		{
	//			dif = middleRadian - (CV_PI / 2);
	//			middleRadian = L1.radian - dif;
	//		}
	//	}
	//	else
	//	{
	//		float dif = (L2.radian - L1.radian + CV_PI) / 2;// (90 - L1.radian) + (90 + L2.radian)
	//		middleRadian = L1.radian + dif;

	//		if (middleRadian > (CV_PI / 2))
	//		{
	//			dif = middleRadian - (CV_PI / 2);
	//			middleRadian = L2.radian - dif;
	//		}
	//	}

	//	drawLine(L1, temp11);
	//	drawLine(L2, temp22);
	//	int a = 1;
	//}
	//else
	//{
	//	middleRadian = (L2.radian + L1.radian) / 2;
	//}

	getMiddleRadian(L1, L2, difRadian, middleRadian);

	L3.radian = middleRadian;
	L3.degree = middleRadian*(180 / CV_PI);
	L3.m = tan(L3.radian);

	if (L1.signMinus /* && L2.signMinus */)
	{
		float x1_OnBottomline;

		if (L1.y1 < L2.y1)
		{
			L3.y1 = L1.y1;

			if (L2.degree != 90)
			{
				x1_OnBottomline = round((L3.y1 - L2.b) / L2.m);
			}
			else
			{
				x1_OnBottomline = L2.x1;
			}

			float totalLength = L1.length + L2.length;
			float ratioL1 = L1.length / totalLength;
			float ratioLine = L2.length / totalLength;

			L3.x1 = round(x1_OnBottomline*ratioLine + L1.x1*ratioL1);
			//L2.x1 = round((x1_OnBottomline + L1.x1) / 2);

		}
		else
		{
			L3.y1 = L2.y1;

			if (L1.degree != 90)
			{
				x1_OnBottomline = round((L3.y1 - L1.b) / L1.m);
			}
			else
			{
				x1_OnBottomline = L1.x1;
			}

			float totalLength = L1.length + L2.length;
			float ratioL1 = L1.length / totalLength;
			float ratioLine = L2.length / totalLength;

			L3.x1 = round(x1_OnBottomline*ratioL1 + L2.x1*ratioLine);
			//L2.x1 = round((x1_OnBottomline + L2.x1) / 2);
		}


		if (L1.y2 > L2.y2)
		{
			L3.y2 = L1.y2;
		}
		else
		{
			L3.y2 = L2.y2;
		}


		if (L3.degree != 90)
		{
			L3.b = L3.y1 - (L3.m * L3.x1);
			L3.x2 = round((L3.y2 - L3.b) / L3.m);
		}
		else
		{
			L3.b = -L3.m;
			L3.x2 = L3.x1;
		}
	}
	else
	{
		float x2_OnBottomline;

		if (L1.y2 < L2.y2)
		{
			L3.y2 = L2.y2;

			if (L2.degree != 90)
			{
				x2_OnBottomline = round((L3.y2 - L2.b) / L2.m);
			}
			else
			{
				x2_OnBottomline = L2.x1;
			}

			float totalLength = L1.length + L2.length;
			float ratioL1 = L1.length / totalLength;
			float ratioLine = L2.length / totalLength;

			L3.x2 = round(x2_OnBottomline*ratioLine + L1.x2*ratioL1);
			//L2.x2 = round((x2_OnBottomline + L1.x2) / 2);
		}
		else
		{
			L3.y2 = L1.y2;

			if (L1.degree != 90)
			{
				x2_OnBottomline = round((L3.y2 - L1.b) / L1.m);
			}
			else
			{
				x2_OnBottomline = L1.x1;
			}

			float totalLength = L1.length + L2.length;
			float ratioL1 = L1.length / totalLength;
			float ratioLine = L2.length / totalLength;

			L3.x2 = round(x2_OnBottomline*ratioL1 + L2.x2*ratioLine);
			//L2.x2 = round((x2_OnBottomline + L2.x2) / 2);
		}

		if (L1.y1 < L2.y1)
		{
			L3.y1 = L1.y1;
		}
		else
		{
			L3.y1 = L2.y1;
		}

		if (L3.degree != 90)
		{
			L3.b = L3.y2 - (L3.m * L3.x2);
			L3.x1 = round((L3.y1 - L3.b) / L3.m);
		}
		else
		{
			L3.b = -L3.m;
			L3.x1 = L3.x2;
		}
	}


	L3.makeline();
	drawLine(L3, temp1);
	drawLine(L3, temp2);

	return L3;
}

void StereoImageBuffer::getBestIndex(std::vector<MyLine>& lineArray, Mat &disparityIMG, float & bestScore, int & bestIndexL1, int & bestIndexL2, float rectangleRatio, float ratioThreshold, int middleY, int banIndexL1, int banIndexL2) const
{
	float length_L1, length_L2;
	float middleX_L1, middleX_L2;
	float vertical_y;//middle y coordinate of the intersected area

	int newsize = lineArray.size();
	float disparityL1;
	float disparityL2;
	const float realWidthDistanceConst = 3.2;//meter
	float left_PointLX, right_PointLX;
	float current_left_PointLX, current_right_PointLX;


	if (banIndexL1 != -1)//(for the second best dock station), the next best dock should be on the right or left side of the first dock detected
	{
		MyLine FirstBest_L1 = lineArray.at(banIndexL1);
		MyLine FirstBest_L2 = lineArray.at(banIndexL2);


		if (FirstBest_L1.x1 < FirstBest_L2.x1)
		{
			left_PointLX = FirstBest_L1.x1;
			right_PointLX = FirstBest_L2.x1;
		}
		else
		{
			left_PointLX = FirstBest_L2.x1;
			right_PointLX = FirstBest_L1.x1;
		}
	}

	for (size_t s = 0; s < newsize; s++)
	{
		MyLine L1 = lineArray.at(s);

		for (size_t o = 0; o < newsize; o++)
		{
			if (o > s && (banIndexL1 != s && banIndexL2 != 0))
			{
				MyLine L2 = lineArray.at(o);

				if (banIndexL1 != -1)//then the second dock should be placed on the left or on the right, not overlapped
				{
					if (L1.x1 < L2.x1)
					{
						current_left_PointLX = L1.x1;
						current_right_PointLX = L2.x1;
					}
					else
					{
						current_left_PointLX = L2.x1;
						current_right_PointLX = L1.x1;
					}

					if (current_left_PointLX < right_PointLX && current_right_PointLX > left_PointLX)
					{
						continue;
					}
				}

				if (!(L1.y2 <= L2.y1 || L1.y1 >= L2.y2)) //if there is intersection
				{
					float totalIntersection;
					if (L1.y1 > L2.y1)
					{
						if (L1.y2 < L2.y2)//short L1 is between L2
						{
							totalIntersection = L1.y2 - L1.y1;
							vertical_y = (L1.y1 + L1.y1) / 2;//middleY of the L1
						}
						else//if (L1.y2 > L2.y2)//L1 is below L2
						{
							totalIntersection = L2.y2 - L1.y1;
							vertical_y = (L1.y1 + L2.y2) / 2;
						}
					}
					else//if if (L1.y1 < L2.y1)
					{
						if (L1.y2 < L2.y2)//L1 is over L2
						{
							totalIntersection = L1.y2 - L2.y1;
							vertical_y = (L2.y1 + L1.y2) / 2;
						}
						else //if (L1.y2 > L2.y2) //short L2 is between L1
						{
							totalIntersection = L2.y2 - L2.y1;
							vertical_y = (L2.y1 + L2.y2) / 2;
						}
					}

					if (L1.degree != 90)
						middleX_L1 = (vertical_y - L1.b) / L1.m;
					else
						middleX_L1 = L1.x1;

					if (L2.degree != 90)
						middleX_L2 = (vertical_y - L2.b) / L2.m;
					else
						middleX_L2 = L2.x1;

					length_L1 = L1.length;
					length_L2 = L2.length;

					float widthDistance = abs(middleX_L1 - middleX_L2);


					float rectRatioL1, rectRatioL2;
					float intersectRatioL1, intersectRatioL2;
					float realWidthDistanceRatio;

					if (widthDistance < length_L1)
						rectRatioL1 = abs(rectangleRatio - (length_L1 / widthDistance));
					else
						rectRatioL1 = abs(rectangleRatio - (widthDistance / length_L1));

					if (widthDistance < length_L2)
						rectRatioL2 = abs(rectangleRatio - (length_L2 / widthDistance));
					else
						rectRatioL2 = abs(rectangleRatio - (widthDistance / length_L2));

					if (totalIntersection == 0)
					{
						intersectRatioL1 = 1;
						intersectRatioL2 = 1;
					}
					else
					{
						intersectRatioL1 = (1 - (totalIntersection / length_L1)) / 2; //divided by 2 since we want rectRatio to be more important
						intersectRatioL2 = (1 - (totalIntersection / length_L2)) / 2;
					}

					//disparity
					//equalizeLines(lineArray.at(o), lineArray.at(s));//first equalize Lines to check real rectangle distances

					int middleY = (int)round(vertical_y);
					int _middleX_L1 = (int)round(middleX_L1);
					int _middleX_L2 = (int)round(middleX_L2);

					disparityL1 = Global_Current_DenseDisparity_Image32F.at<float>(middleY, _middleX_L1);
					disparityL2 = Global_Current_DenseDisparity_Image32F.at<float>(middleY, _middleX_L2);
					int difDisp = abs(disparityL1 - disparityL2);

					MyDockObject *objL1, *objL2, *objCommon;
					float realWidthDistance, realBottomHeight;

					if ((difDisp < 3) && (disparityL1 < 0xFFF0 && disparityL1 > 0) && (disparityL2 < 0xFFF0 && disparityL2 > 0))
					{

						objL1 = new MyDockObject(disparityL1, _middleX_L1, middleY);
						ImageToWorldCoordinates(*objL1);

						objL2 = new MyDockObject(disparityL2, _middleX_L2, middleY);
						ImageToWorldCoordinates(*objL2);


						realWidthDistance = abs(objL2->Xw - objL1->Xw);

						float midDisp = (disparityL2 + disparityL1) / 2;
						int midX = (int)round((middleX_L1 + _middleX_L2) / 2);

						objCommon = new MyDockObject(midDisp, midX, middleY);
						ImageToWorldCoordinates(*objCommon);
						realBottomHeight = objCommon->Yw;
					}
					else
					{
						realWidthDistance = 0.0;
						realBottomHeight = 0.0;
					}
					//disparity

					realWidthDistanceRatio = abs(1 - (realWidthDistance / realWidthDistanceConst));

					float ratio = (((rectRatioL1 + rectRatioL2 + intersectRatioL1 + intersectRatioL2) / 4) + realWidthDistanceRatio) / 2;

					cout << s << "-" << o << fixed << std::setprecision(2) << " Rdist:" << realWidthDistance << " Rheight:" << realBottomHeight << " rR:" << realWidthDistanceRatio << " dist:" << widthDistance << " rL1:" << rectRatioL1 << " rL2:" << rectRatioL2 << " iL1:" << intersectRatioL1 << " iL2:" << intersectRatioL2 << " R:" << ratio << endl;

					if (ratio <= ratioThreshold && ratio < bestScore)
					{
						bestScore = ratio;
						bestIndexL2 = o;
						bestIndexL1 = s;
						lineArray.at(s).wallRatio = rectRatioL1;
						lineArray.at(o).wallRatio = rectRatioL2;
						lineArray.at(s).realwidthDistance = realWidthDistance;
						lineArray.at(o).realHeight = realBottomHeight;
					}


				}//if there is intersection
			}//if not the same object
		}//for loop L2
	}//for loop L1
}

inline void StereoImageBuffer::equalizeLines(MyLine &line1, MyLine &line2) const
{
	if (line1.wallRatio == -1 || line2.wallRatio == -1)
		return;

	//put line's bottom corrdinates to floor
	float Yw_bottom = 0.0;
	float Yw_up = 4.25;

	float Zs_L1 = (stereoParams.fx * stereoParams.b) / line1.d;
	float Zs_L2 = (stereoParams.fx * stereoParams.b) / line2.d;

	float Ysb_L1 = (Yw_bottom + Zs_L1*sin(stereoParams.tilt) - stereoParams.cY) / cos(stereoParams.tilt);
	float Ysb_L2 = (Yw_bottom + Zs_L2*sin(stereoParams.tilt) - stereoParams.cY) / cos(stereoParams.tilt);
	float Ysu_L1 = (Yw_up + Zs_L1*sin(stereoParams.tilt) - stereoParams.cY) / cos(stereoParams.tilt);
	float Ysu_L2 = (Yw_up + Zs_L2*sin(stereoParams.tilt) - stereoParams.cY) / cos(stereoParams.tilt);

	line1.realBottom_y = (int)round(stereoParams.v0 - stereoParams.fy * (Ysb_L1 / Zs_L1));
	line2.realBottom_y = (int)round(stereoParams.v0 - stereoParams.fy * (Ysb_L2 / Zs_L2));

	line1.realUp_y = (int)round(stereoParams.v0 - stereoParams.fy * (Ysu_L1 / Zs_L1));
	line2.realUp_y = (int)round(stereoParams.v0 - stereoParams.fy * (Ysu_L2 / Zs_L2));

	//line1.realBottom_y = get_v_From_VDisparity_RoadProfile(line1.d);
	//line2.realBottom_y = get_v_From_VDisparity_RoadProfile(line2.d);

	if (!isinf(line1.m))
	{
		line1.realBottom_x = round((line1.realBottom_y - line1.b) / line1.m);
		line1.realUp_x = round((line1.realUp_y - line1.b) / line1.m);
	}
	else
	{
		line1.realUp_x = line1.x1;
		line1.realBottom_x = line1.x2;
	}

	if (!isinf(line2.m))
	{
		line2.realBottom_x = round((line2.realBottom_y - line2.b) / line2.m);
		line2.realUp_x = round((line2.realUp_y - line2.b) / line2.m);
	}
	else
	{
		line2.realUp_x = line2.x1;
		line2.realBottom_x = line2.x2;
	}

	if (line1.wallRatio < line2.wallRatio)
	{
		if (line2.degree != 90)//if not 90 degree use the line equation
		{
			line2.y1 = line1.y1;
			line2.x1 = round((line1.y1 - line2.b) / line2.m);

			line2.y2 = line1.y2;
			line2.x2 = round((line1.y2 - line2.b) / line2.m);
		}
		else
		{
			line2.y1 = line1.y1;
			line2.y2 = line1.y2;
		}

		line2.makeline();
	}
	else
	{
		if (line1.degree != 90)//if not 90 degree use the line equation
		{
			line1.y1 = line2.y1;
			line1.x1 = round((line2.y1 - line1.b) / line1.m);

			line1.y2 = line2.y2;
			line1.x2 = round((line2.y2 - line1.b) / line1.m);
		}
		else
		{
			line1.y1 = line2.y1;
			line1.y2 = line2.y2;
		}

		line1.makeline();
	}

}

inline void  StereoImageBuffer::drawLine(float angle, float startX, float startY, float endY, Mat &draw) const
{
	float deg = angle;
	float rad = deg*(CV_PI / 180);
	float m = tan(rad);
	float n = startY - (m * startX);
	float endX = (endY - n) / m;

	int r = rand() % 255;
	int g = rand() % 255;
	int b = rand() % 255;

	cv::line(draw, Point(startX, startY), Point(endX, endY), Scalar(r, g, b), 1, cv::LineTypes::LINE_4);

	std::ostringstream text;


	text << "deg:" << deg << " x1 " << startX << "," << startY;
	if (signbit(deg) == 1)
	{
		cv::putText(draw, text.str(), Point(startX + 25, startY), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(r, g, b), 1, CV_AA);
		text.str("");
		text.clear();

		text << "deg:" << deg << " x2 " << endX << "," << endY;
		cv::putText(draw, text.str(), Point(endX + 25, endY), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(r, g, b), 1, CV_AA);
		text.str("");
		text.clear();
	}
	else
	{
		cv::putText(draw, text.str(), Point(startX - 250, startY), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(r, g, b), 1, CV_AA);
		text.str("");
		text.clear();

		text << "deg:" << deg << " x2 " << endX << "," << endY;
		cv::putText(draw, text.str(), Point(endX - 250, endY), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(r, g, b), 1, CV_AA);
		text.str("");
		text.clear();
	}

}

inline void StereoImageBuffer::drawLine(MyLine myline, Mat& img) const
{
	int r = rand() % 255;
	int g = rand() % 255;
	int b = rand() % 255;

	cv::line(img, Point(myline.x1, myline.y1), Point(myline.x2, myline.y2), Scalar(r, g, b), 1, cv::LineTypes::LINE_4);
	int x1, x2, y1, y2;

	if ((int)myline.x1 % 3 == 0)
	{
		x1 = (int)myline.x1 + 10;
		x2 = (int)myline.x2 - 80;
		y1 = (int)myline.y1;
		y2 = (int)myline.y2;
	}

	if ((int)myline.x1 % 3 == 1)
	{
		x1 = (int)myline.x1 - 80;
		x2 = (int)myline.x2 + 10;
		y1 = (int)myline.y1;
		y2 = (int)myline.y2;
	}

	if ((int)myline.x1 % 3 == 2)
	{
		x1 = (int)myline.x1 - 80;
		x2 = (int)myline.x2 + 10;
		y1 = (int)((myline.y1 + myline.y2) / 2);
		y2 = (int)myline.y1 + 30;
	}


	std::ostringstream text;
	text << myline.index << ":" << (int)round(myline.degree) << " " << myline.x1 << "," << myline.y1;
	cv::putText(img, text.str(), Point(x1, y1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(r, g, b), 1, CV_AA);
	text.str("");
	text.clear();

	text << myline.index << ":" << (int)round(myline.degree) << " " << myline.x2 << "," << myline.y2;
	cv::putText(img, text.str(), Point(x2 - 20, y2), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(r, g, b), 1, CV_AA);
	text.str("");
	text.clear();
}

inline void StereoImageBuffer::drawLine(Vec4i line, Mat& img) const
{
	int x1 = line[0];
	int y1 = line[1];
	int x2 = line[2];
	int y2 = line[3];

	int r = rand() % 255;
	int g = rand() % 255;
	int b = rand() % 255;

	cv::line(img, Point(x1, y1), Point(x2, y2), Scalar(r, g, b), 1, cv::LineTypes::LINE_4);

	float radian;
	radian = std::atan2((y2 - y1), (x2 - x1));
	//radian = std::atan((y2 - y1) / (x2 - x1));
	float degree = radian*(180 / CV_PI);
	float length = sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));

	std::ostringstream text;
	text << 0 << ":" << degree << " x1 " << x1 << "," << y1;
	cv::putText(img, text.str(), Point(x1 + 50, y1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(r, g, b), 1, CV_AA);
	text.str("");
	text.clear();

	text << 0 << ":" << degree << " x2 " << x2 << "," << y2;
	cv::putText(img, text.str(), Point(x2 - 200, y2), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(r, g, b), 1, CV_AA);
	text.str("");
	text.clear();
}

std::vector<StereoImageBuffer::MyLine> StereoImageBuffer::filterAndMakeLines(std::vector<cv::Vec4i> allLines, float minLength, float minDegree, Mat& imgTemp) const
{
	int totalNumOfLines = allLines.size();
	std::vector<MyLine> filteredLines;

	Mat temp1, temp2;
	temp1.create(imgTemp.size(), imgTemp.type());
	temp2.create(imgTemp.size(), imgTemp.type());


	if (!totalNumOfLines <= 0) //if there are lines/line
	{
		int lineIndex = 0;
		for (size_t i = 0; i < totalNumOfLines; i++)
		{
			temp1 = Scalar(0, 0, 0);
			temp2 = Scalar(0, 0, 0);

			drawLine(allLines[i], temp1);
			MyLine line(allLines[i]);

			line.index = 0;
			drawLine(line, temp2);

			if (fabs(line.degree) > minDegree && line.length > minLength)
			{
				line.index = lineIndex;
				filteredLines.push_back(line);
				lineIndex++;
			}

			//cout << fixed << std::setprecision(2) << i << ". Deg:" << line.degree << std::setprecision(0) << " --> Len: " << line.length << " x1:" << line.x1 << " y1:" << line.y1 << " x2:" << line.x2 << " y2:" << line.y2 << endl;

			//else
			//{
			//	drawLine(line, imgTemp);
			//	cout << "";
			//}
		}

		return filteredLines;
	}
}

int StereoImageBuffer::mergeLines(std::vector<cv::Vec4i> allLines, Mat& img, Mat& imgOrg, Mat& disparityIMG, std::vector<MyLine> &rectangle1, std::vector<MyLine> &rectangle2, Mat& occupancyMatrix, Mat& line_draws) const
{
	float minlen = 25;
	float minDegree = 70;


	//display, delete later
	float middleX = round(img.cols / 2);
	float middleY = round(img.rows / 2);
	Mat img1, imgTemp, imgTemp1, imgTemp2, imgTemp3, imgOrg2;
	img.copyTo(img1);
	img.copyTo(imgTemp);
	imgTemp1.create(img.size(), img.type());
	imgTemp2.create(img.size(), img.type());
	imgTemp3.create(img.size(), img.type());
	imgOrg.copyTo(imgOrg2);
	//display, delete later

	std::vector<MyLine> allmyLines = filterAndMakeLines(allLines, minlen, minDegree, imgTemp);
	int totalNumOfLines = allmyLines.size();

	//Alternative 
	//std::vector<int> labels;
	//int numberOfLines = cv::partition(allLines, labels, isEqual);
	//Alternative

	if (!totalNumOfLines <= 0) //if there are lines/line
	{
		std::vector<double> degreeArray;
		std::vector<int> lineArrayCount;
		std::vector<MyLine> lineArray;

		double currentLine_m;
		double currentLineAngleRadian;
		double currentLineAngleDegree;
		double currentLineLength = 0;

		double average_Length = 0;

		float degreePrecision = 3;
		float coordinatePrecision = 12;

		for (size_t i = 0; i < totalNumOfLines; i++)
		{
			//float x1 = allLines[i][0];
			//float y1 = allLines[i][1];
			//float x2 = allLines[i][2];
			//float y2 = allLines[i][3];

			//if (x2 != x1)
			//	currentLine_m = ((y2 - y1) / (x2 - x1));
			//else
			//	currentLine_m = -1;

			//currentLineAngleRadian = std::atan2((y2 - y1), (x2 - x1));
			//currentLineAngleDegree = ceil(currentLineAngleRadian*(180 / CV_PI));
			//currentLineLength = sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));
			////cout << fixed << std::setprecision(2) << i << ". Degree:" << currentLineAngleDegree << std::setprecision(0) << " --> Length: " << currentLineLength << " x1:" << x1 << " y1:" << y1 << " x2:" << x2 << " y2:" << y2 << endl;

			//average_Length += currentLineLength;			

			//MyLine currentLine;
			//currentLine.x1 = x1;
			//currentLine.y1 = y1;
			//currentLine.x2 = x2;
			//currentLine.y2 = y2;
			//currentLine.makeline();
			//currentLine.index = i;

			MyLine currentLine;
			currentLine = allmyLines[i];
			currentLine.index = i;

			drawLine(currentLine, imgTemp1);

			//if (!(abs(currentLine.degree) > 80 && abs(currentLine.degree) < 100))
			//{
			//	continue;
			//}

			int lSize = lineArray.size();
			if (lSize > 0)
			{
				bool found = false;
				for (size_t l = 0; l < lSize; l++)
				{
					float deg = lineArray.at(l).degree;
					float cx1 = lineArray.at(l).x1;
					float cx2 = lineArray.at(l).x2;
					float cy1 = lineArray.at(l).y1;
					float cy2 = lineArray.at(l).y2;
					float cm = lineArray.at(l).m;
					float cb = lineArray.at(l).b;
					float len = lineArray.at(l).length;

					float difRadian, difDegree;
					getAngleBetweenlines(currentLine, lineArray.at(l), difRadian, difDegree);

					if (difDegree < degreePrecision)
					{
						float currentLineMiddleX = 0;

						if (currentLine.degree != 90)
						{
							currentLineMiddleX = (middleY - currentLine.b) / currentLine.m; //middle row point of the line
						}
						else
						{
							currentLineMiddleX = currentLine.x1;
						}

						float lineMiddleX = 0;
						if (deg != 90)
						{
							lineMiddleX = (middleY - cb) / cm; //middle row point of the line
						}
						else
						{
							lineMiddleX = cx1;
						}

						//calculate the distance betweem the middle points
						float distance;
						distance = sqrt(pow((currentLineMiddleX - lineMiddleX), 2));

						//float mx1 = (currentLine.x1 + cx1) * 0.5f;// l1x1 + l1x2
						//float mx2 = (currentLine.x2 + cx2) * 0.5f;// l2x1 + l2x2
						//float my1 = (currentLine.y1 + cy1) * 0.5f;// l1y1 + l1y2
						//float my2 = (currentLine.y2 + cy2)* 0.5f;// l2y1 + l2y2
						//float dist = sqrtf((mx1 - mx2)*(mx1 - mx2) + (my1 - my2)*(my1 - my2));

						//float lenScore = std::max(currentLine.length, len) * 0.5f;

						//if (dist < lenScore) cout << dist << " < " << lenScore << " - ";
						//else cout << dist << " > " << lenScore << " - ";

						//if (distance < coordinatePrecision) cout << distance << " < " << coordinatePrecision << endl;
						//else cout << distance << " > " << coordinatePrecision << endl;


						if (distance < coordinatePrecision)//if (dist < lenScore)						
						{
							found = true;
							lineArrayCount.at(l) = lineArrayCount.at(l) + 1;
							lineArray.at(l) = merge2Lines(currentLine, lineArray.at(l));
							lineArray.at(l).index = l;
							drawLine(lineArray.at(l), imgTemp2);
							break;
						}

					}
				}

				if (!found)
				{
					lineArray.push_back(currentLine);
					lineArrayCount.push_back(1);

					int currentIndex = lineArray.size() - 1;
					currentLine.index = currentIndex;
					drawLine(currentLine, imgTemp2);
					cout << "";
				}
			}
			else
			{
				lineArray.push_back(currentLine);
				lineArrayCount.push_back(1);

				currentLine.index = lineArray.size() - 1;
				drawLine(currentLine, imgTemp2);
				cout << "";
			}

		}

		std::cout << endl << endl;
		for (size_t deg = 0; deg < lineArray.size(); deg++)
		{
			lineArray.at(deg).index = deg;
			drawLine(lineArray.at(deg), imgTemp3);
			cout << "";
			//cout << fixed << std::setprecision(2) << deg << ". Degree:" << lineArray.at(deg).degree << std::setprecision(0) << " Count: " << lineArrayCount.at(deg) << " --> Length: " << lineArray.at(deg).length << " x1:" << lineArray.at(deg).x1 << " y1:" << lineArray.at(deg).y1 << " x2:" << lineArray.at(deg).x2 << " y2:" << lineArray.at(deg).y2 << endl;
		}

		//Select the best 2 lines
		int scoreIndex_o = -1;
		int scoreIndex_s = -1;
		int scoreIndex2_o = -1;
		int scoreIndex2_s = -1;

		float rectangleRatio = 1.125; // height / width
		float ratioThreshold = 0.14; // height / width //--> 0.1 is the best

		float bestScor = 1000000;
		float bestScor2 = 1000000;

		//cv::sortIdx(source, dst, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
		clock_t start;
		clock_t end;

		//start = clock();
		getBestIndex(lineArray, disparityIMG, bestScor, scoreIndex_s, scoreIndex_o, rectangleRatio, ratioThreshold, middleY, -1, -1);
		cout << endl << endl;
		getBestIndex(lineArray, disparityIMG, bestScor2, scoreIndex2_s, scoreIndex2_o, rectangleRatio, ratioThreshold, middleY, scoreIndex_s, scoreIndex_o);
		//end = clock();

		//std::cout << lineArray.size() << " lines: getBestIndex X 2 " << fixed << std::setprecision(4) << float(end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;

		Mat draw1, draw2;
		draw1.create(img.size(), img.type());
		draw2.create(img.size(), img.type());

		line_draws = imgTemp3;

		cvtColor(occupancyMatrix, occupancyMatrix, CV_GRAY2BGR);
		if (scoreIndex_o != -1)
		{
			//float disparityL1 = (float)disparityIMG.at<uchar>(lineArray.at(scoreIndex_o).y1, lineArray.at(scoreIndex_o).x1);
			//float disparityL2 = (float)disparityIMG.at<uchar>(lineArray.at(scoreIndex_s).y1, lineArray.at(scoreIndex_s).x1);

			float disparityL1 = Global_Current_DenseDisparity_Image32F.at<float>(lineArray.at(scoreIndex_o).y2, lineArray.at(scoreIndex_o).x2);
			float disparityL2 = Global_Current_DenseDisparity_Image32F.at<float>(lineArray.at(scoreIndex_s).y2, lineArray.at(scoreIndex_s).x2);
			lineArray.at(scoreIndex_o).d = disparityL1;
			lineArray.at(scoreIndex_s).d = disparityL2;

			MyDockObject *objL1, *objL2;
			objL1 = new MyDockObject(disparityL1, lineArray.at(scoreIndex_o).x2, lineArray.at(scoreIndex_o).y2);
			objL2 = new MyDockObject(disparityL2, lineArray.at(scoreIndex_s).x2, lineArray.at(scoreIndex_s).y2);
			ImageToWorldCoordinates(*objL1);
			ImageToWorldCoordinates(*objL2);

			cout << "Best Score1: (" << scoreIndex_s << "," << scoreIndex_o << ") = " << fixed << std::setprecision(2) << bestScor << " Real Distance:" << lineArray.at(scoreIndex_s).realwidthDistance << "-" << lineArray.at(scoreIndex_o).realwidthDistance << " Real Height:" << lineArray.at(scoreIndex_s).realHeight << endl;
			cout << fixed << std::setprecision(2) << "L1_X:" << objL1->Xw << " L1_Z:" << objL1->Zw << " L1_Y:" << objL1->Yw << " L2_X:" << objL2->Xw << " L2_Z:" << objL2->Zw << " L2_Y:" << objL2->Yw << " Length:" << abs(objL2->Xw - objL1->Xw) << endl;

			int Xw1 = (int)round((10 + objL1->Xw) * 10);
			int Zw1 = (int)round((400 - objL1->Zw * 10));
			int Xw2 = (int)round((10 + objL2->Xw) * 10);
			int Zw2 = (int)round((400 - objL2->Zw * 10));

			cv::line(occupancyMatrix, Point(Xw1, Zw1), Point(Xw2, Zw2), Scalar(255, 0, 255), 2, cv::LineTypes::LINE_4);

			//cout << "Best Score: " << scoreIndex_s << "-" << scoreIndex_o << endl;
			cv::line(draw1, Point(lineArray.at(scoreIndex_o).x1, lineArray.at(scoreIndex_o).y1), Point(lineArray.at(scoreIndex_o).x2, lineArray.at(scoreIndex_o).y2), Scalar(0, 0, 255), 2, cv::LineTypes::LINE_4);
			cv::line(draw1, Point(lineArray.at(scoreIndex_s).x1, lineArray.at(scoreIndex_s).y1), Point(lineArray.at(scoreIndex_s).x2, lineArray.at(scoreIndex_s).y2), Scalar(255, 0, 0), 2, cv::LineTypes::LINE_4);
			cv::line(imgOrg, Point(lineArray.at(scoreIndex_o).x1, lineArray.at(scoreIndex_o).y1), Point(lineArray.at(scoreIndex_o).x2, lineArray.at(scoreIndex_o).y2), Scalar(0, 0, 255), 10, cv::LineTypes::LINE_4);
			cv::line(imgOrg, Point(lineArray.at(scoreIndex_s).x1, lineArray.at(scoreIndex_s).y1), Point(lineArray.at(scoreIndex_s).x2, lineArray.at(scoreIndex_s).y2), Scalar(255, 0, 0), 10, cv::LineTypes::LINE_4);


			equalizeLines(lineArray.at(scoreIndex_o), lineArray.at(scoreIndex_s));

			cv::line(draw2, Point(lineArray.at(scoreIndex_o).realUp_x, lineArray.at(scoreIndex_o).realUp_y), Point(lineArray.at(scoreIndex_o).realBottom_x, lineArray.at(scoreIndex_o).realBottom_y), Scalar(0, 100, 255), 2, cv::LineTypes::LINE_4);
			cv::line(draw2, Point(lineArray.at(scoreIndex_s).realUp_x, lineArray.at(scoreIndex_s).realUp_y), Point(lineArray.at(scoreIndex_s).realBottom_x, lineArray.at(scoreIndex_s).realBottom_y), Scalar(255, 100, 0), 2, cv::LineTypes::LINE_4);

			std::ostringstream text;

			cv::line(imgOrg2, Point(lineArray.at(scoreIndex_o).x1, lineArray.at(scoreIndex_o).y1), Point(lineArray.at(scoreIndex_o).x2, lineArray.at(scoreIndex_o).y2), Scalar(0, 0, 255), 2, cv::LineTypes::LINE_4);
			cv::line(imgOrg2, Point(lineArray.at(scoreIndex_o).realUp_x, lineArray.at(scoreIndex_o).realUp_y), Point(lineArray.at(scoreIndex_o).realBottom_x, lineArray.at(scoreIndex_o).realBottom_y), Scalar(0, 100, 255), 2, cv::LineTypes::LINE_4);

			text << lineArray.at(scoreIndex_o).degree << " x1 " << lineArray.at(scoreIndex_o).x1 << "," << lineArray.at(scoreIndex_o).y1;
			cv::putText(imgOrg2, text.str(), Point(lineArray.at(scoreIndex_o).x1 + 50, lineArray.at(scoreIndex_o).y1), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 1, CV_AA);
			text.str("");
			text.clear();

			text << lineArray.at(scoreIndex_o).degree << " x2 " << lineArray.at(scoreIndex_o).x2 << "," << lineArray.at(scoreIndex_o).y2;
			cv::putText(imgOrg2, text.str(), Point(lineArray.at(scoreIndex_o).x2 - 200, lineArray.at(scoreIndex_o).y2), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 1, CV_AA);
			text.str("");
			text.clear();

			cv::line(imgOrg2, Point(lineArray.at(scoreIndex_s).x1, lineArray.at(scoreIndex_s).y1), Point(lineArray.at(scoreIndex_s).x2, lineArray.at(scoreIndex_s).y2), Scalar(255, 0, 0), 2, cv::LineTypes::LINE_4);
			cv::line(imgOrg2, Point(lineArray.at(scoreIndex_s).realUp_x, lineArray.at(scoreIndex_s).realUp_y), Point(lineArray.at(scoreIndex_s).realBottom_x, lineArray.at(scoreIndex_s).realBottom_y), Scalar(255, 100, 0), 2, cv::LineTypes::LINE_4);

			text << lineArray.at(scoreIndex_s).degree << " x1 " << lineArray.at(scoreIndex_s).x1 << "," << lineArray.at(scoreIndex_s).y1;
			cv::putText(imgOrg2, text.str(), Point(lineArray.at(scoreIndex_s).x1 + 50, lineArray.at(scoreIndex_s).y1), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0), 1, CV_AA);
			text.str("");
			text.clear();

			text << lineArray.at(scoreIndex_s).degree << " x2 " << lineArray.at(scoreIndex_s).x2 << "," << lineArray.at(scoreIndex_s).y2;
			cv::putText(imgOrg2, text.str(), Point(lineArray.at(scoreIndex_s).x2 - 200, lineArray.at(scoreIndex_s).y2), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0), 1, CV_AA);
			text.str("");
			text.clear();

			rectangle1.push_back(lineArray.at(scoreIndex_o));
			rectangle1.push_back(lineArray.at(scoreIndex_s));

			MyLine bottomRow, topRow;
			bottomRow.x1 = lineArray.at(scoreIndex_o).realBottom_x;
			bottomRow.y1 = lineArray.at(scoreIndex_o).realBottom_y;
			bottomRow.x2 = lineArray.at(scoreIndex_s).realBottom_x;
			bottomRow.y2 = lineArray.at(scoreIndex_s).realBottom_y;

			topRow.x1 = lineArray.at(scoreIndex_o).realUp_x;
			topRow.y1 = lineArray.at(scoreIndex_o).realUp_y;
			topRow.x2 = lineArray.at(scoreIndex_s).realUp_x;
			topRow.y2 = lineArray.at(scoreIndex_s).realUp_y;

			cv::line(imgOrg2, Point(bottomRow.x1, bottomRow.y1), Point(bottomRow.x2, bottomRow.y2), Scalar(0, 0, 255), 2, cv::LineTypes::LINE_4);
			cv::line(imgOrg2, Point(topRow.x1, topRow.y1), Point(topRow.x2, topRow.y2), Scalar(255, 0, 0), 2, cv::LineTypes::LINE_4);
			cv::line(draw1, Point(lineArray.at(scoreIndex_o).x1, lineArray.at(scoreIndex_o).y1), Point(lineArray.at(scoreIndex_s).x1, lineArray.at(scoreIndex_s).y1), Scalar(0, 0, 255), 2, cv::LineTypes::LINE_4);
			cv::line(draw1, Point(lineArray.at(scoreIndex_o).x2, lineArray.at(scoreIndex_o).y2), Point(lineArray.at(scoreIndex_s).x2, lineArray.at(scoreIndex_s).y2), Scalar(0, 0, 255), 2, cv::LineTypes::LINE_4);

			cv::line(draw2, Point(bottomRow.x1, bottomRow.y1), Point(bottomRow.x2, bottomRow.y2), Scalar(0, 0, 255), 2, cv::LineTypes::LINE_4);
			cv::line(draw2, Point(topRow.x1, topRow.y1), Point(topRow.x2, topRow.y2), Scalar(255, 0, 0), 2, cv::LineTypes::LINE_4);

			rectangle1.push_back(bottomRow);
			rectangle1.push_back(topRow);
		}//if (scoreIndex_o != -1)

		if (scoreIndex2_o != -1 && scoreIndex_o != scoreIndex2_o && scoreIndex_s != scoreIndex2_s)
		{
			//float disparityL1 = (float)disparityIMG.at<uchar>(lineArray.at(scoreIndex2_o).y1, lineArray.at(scoreIndex2_o).x1);
			//float disparityL2 = (float)disparityIMG.at<uchar>(lineArray.at(scoreIndex2_s).y1, lineArray.at(scoreIndex2_s).x1);

			float disparityL1 = Global_Current_DenseDisparity_Image32F.at<float>(lineArray.at(scoreIndex2_o).y2, lineArray.at(scoreIndex2_o).x2);
			float disparityL2 = Global_Current_DenseDisparity_Image32F.at<float>(lineArray.at(scoreIndex2_s).y2, lineArray.at(scoreIndex2_s).x2);
			lineArray.at(scoreIndex2_o).d = disparityL1;
			lineArray.at(scoreIndex2_s).d = disparityL2;

			MyDockObject *objL1, *objL2;
			objL1 = new MyDockObject(disparityL1, lineArray.at(scoreIndex2_o).x2, lineArray.at(scoreIndex2_o).y2);
			objL2 = new MyDockObject(disparityL2, lineArray.at(scoreIndex2_s).x2, lineArray.at(scoreIndex2_s).y2);
			ImageToWorldCoordinates(*objL1);
			ImageToWorldCoordinates(*objL2);


			cout << "Best Score2: (" << scoreIndex2_s << "," << scoreIndex2_o << ") = " << fixed << std::setprecision(2) << bestScor << " Real Distance:" << lineArray.at(scoreIndex_s).realwidthDistance << "-" << lineArray.at(scoreIndex_o).realwidthDistance << " Real Height:" << lineArray.at(scoreIndex_s).realHeight << endl;
			cout << fixed << std::setprecision(2) << "L1_X:" << objL1->Xw << " L1_Z:" << objL1->Zw << " L1_Y:" << objL1->Yw << " L2_X:" << objL2->Xw << " L2_Z:" << objL2->Zw << " L2_Y:" << objL2->Yw << " Length:" << abs(objL2->Xw - objL1->Xw) << endl;

			int Xw1 = (int)round((10 + objL1->Xw) * 10);
			int Zw1 = (int)round((400 - objL1->Zw * 10));
			int Xw2 = (int)round((10 + objL2->Xw) * 10);
			int Zw2 = (int)round((400 - objL2->Zw * 10));

			cv::line(occupancyMatrix, Point(Xw1, Zw1), Point(Xw2, Zw2), Scalar(255, 0, 255), 2, cv::LineTypes::LINE_4);

			//cout << "Best Score: " << scoreIndex2_s << "-" << scoreIndex2_o << endl;
			cv::line(draw1, Point(lineArray.at(scoreIndex2_o).x1, lineArray.at(scoreIndex2_o).y1), Point(lineArray.at(scoreIndex2_o).x2, lineArray.at(scoreIndex2_o).y2), Scalar(0, 0, 255), 2, cv::LineTypes::LINE_4);
			cv::line(draw1, Point(lineArray.at(scoreIndex2_s).x1, lineArray.at(scoreIndex2_s).y1), Point(lineArray.at(scoreIndex2_s).x2, lineArray.at(scoreIndex2_s).y2), Scalar(255, 0, 0), 2, cv::LineTypes::LINE_4);
			cv::line(imgOrg, Point(lineArray.at(scoreIndex2_o).x1, lineArray.at(scoreIndex2_o).y1), Point(lineArray.at(scoreIndex2_o).x2, lineArray.at(scoreIndex2_o).y2), Scalar(0, 0, 255), 10, cv::LineTypes::LINE_4);
			cv::line(imgOrg, Point(lineArray.at(scoreIndex2_s).x1, lineArray.at(scoreIndex2_s).y1), Point(lineArray.at(scoreIndex2_s).x2, lineArray.at(scoreIndex2_s).y2), Scalar(255, 0, 0), 10, cv::LineTypes::LINE_4);


			equalizeLines(lineArray.at(scoreIndex2_o), lineArray.at(scoreIndex2_s));

			cv::line(draw2, Point(lineArray.at(scoreIndex2_o).realUp_x, lineArray.at(scoreIndex2_o).realUp_y), Point(lineArray.at(scoreIndex2_o).realBottom_x, lineArray.at(scoreIndex2_o).realBottom_y), Scalar(0, 100, 255), 2, cv::LineTypes::LINE_4);
			cv::line(draw2, Point(lineArray.at(scoreIndex2_s).realUp_x, lineArray.at(scoreIndex2_s).realUp_y), Point(lineArray.at(scoreIndex2_s).realBottom_x, lineArray.at(scoreIndex2_s).realBottom_y), Scalar(255, 100, 0), 2, cv::LineTypes::LINE_4);

			std::ostringstream text;

			cv::line(imgOrg2, Point(lineArray.at(scoreIndex2_o).x1, lineArray.at(scoreIndex2_o).y1), Point(lineArray.at(scoreIndex2_o).x2, lineArray.at(scoreIndex2_o).y2), Scalar(0, 0, 255), 2, cv::LineTypes::LINE_4);
			cv::line(imgOrg2, Point(lineArray.at(scoreIndex2_o).realUp_x, lineArray.at(scoreIndex2_o).realUp_y), Point(lineArray.at(scoreIndex2_o).realBottom_x, lineArray.at(scoreIndex2_o).realBottom_y), Scalar(0, 0, 255), 2, cv::LineTypes::LINE_4);

			text << lineArray.at(scoreIndex2_o).degree << " x1 " << lineArray.at(scoreIndex2_o).x1 << "," << lineArray.at(scoreIndex2_o).y1;
			cv::putText(imgOrg2, text.str(), Point(lineArray.at(scoreIndex2_o).x1 + 50, lineArray.at(scoreIndex2_o).y1), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 1, CV_AA);
			text.str("");
			text.clear();

			text << lineArray.at(scoreIndex2_o).degree << " x2 " << lineArray.at(scoreIndex2_o).x2 << "," << lineArray.at(scoreIndex2_o).y2;
			cv::putText(imgOrg2, text.str(), Point(lineArray.at(scoreIndex2_o).x2 - 200, lineArray.at(scoreIndex2_o).y2), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 1, CV_AA);
			text.str("");
			text.clear();

			cv::line(imgOrg2, Point(lineArray.at(scoreIndex2_s).x1, lineArray.at(scoreIndex2_s).y1), Point(lineArray.at(scoreIndex2_s).x2, lineArray.at(scoreIndex2_s).y2), Scalar(255, 0, 0), 2, cv::LineTypes::LINE_4);
			cv::line(imgOrg2, Point(lineArray.at(scoreIndex2_s).realUp_x, lineArray.at(scoreIndex2_s).realUp_y), Point(lineArray.at(scoreIndex2_s).realBottom_x, lineArray.at(scoreIndex2_s).realBottom_y), Scalar(255, 0, 0), 2, cv::LineTypes::LINE_4);

			text << lineArray.at(scoreIndex2_s).degree << " x1 " << lineArray.at(scoreIndex2_s).x1 << "," << lineArray.at(scoreIndex2_s).y1;
			cv::putText(imgOrg2, text.str(), Point(lineArray.at(scoreIndex2_s).x1 + 50, lineArray.at(scoreIndex2_s).y1), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0), 1, CV_AA);
			text.str("");
			text.clear();

			text << lineArray.at(scoreIndex2_s).degree << " x2 " << lineArray.at(scoreIndex2_s).x2 << "," << lineArray.at(scoreIndex2_s).y2;
			cv::putText(imgOrg2, text.str(), Point(lineArray.at(scoreIndex2_s).x2 - 200, lineArray.at(scoreIndex2_s).y2), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0), 1, CV_AA);
			text.str("");
			text.clear();

			rectangle2.push_back(lineArray.at(scoreIndex2_o));
			rectangle2.push_back(lineArray.at(scoreIndex2_s));

			MyLine bottomRow, topRow;
			bottomRow.x1 = lineArray.at(scoreIndex2_o).realBottom_x;
			bottomRow.y1 = lineArray.at(scoreIndex2_o).realBottom_y;
			bottomRow.x2 = lineArray.at(scoreIndex2_s).realBottom_x;
			bottomRow.y2 = lineArray.at(scoreIndex2_s).realBottom_y;

			topRow.x1 = lineArray.at(scoreIndex2_o).realUp_x;
			topRow.y1 = lineArray.at(scoreIndex2_o).realUp_y;
			topRow.x2 = lineArray.at(scoreIndex2_s).realUp_x;
			topRow.y2 = lineArray.at(scoreIndex2_s).realUp_y;

			cv::line(imgOrg2, Point(bottomRow.x1, bottomRow.y1), Point(bottomRow.x2, bottomRow.y2), Scalar(0, 0, 255), 2, cv::LineTypes::LINE_4);
			cv::line(imgOrg2, Point(topRow.x1, topRow.y1), Point(topRow.x2, topRow.y2), Scalar(255, 0, 0), 2, cv::LineTypes::LINE_4);
			cv::line(draw1, Point(lineArray.at(scoreIndex2_o).x1, lineArray.at(scoreIndex2_o).y1), Point(lineArray.at(scoreIndex2_s).x1, lineArray.at(scoreIndex2_s).y1), Scalar(0, 0, 255), 2, cv::LineTypes::LINE_4);
			cv::line(draw1, Point(lineArray.at(scoreIndex2_o).x2, lineArray.at(scoreIndex2_o).y2), Point(lineArray.at(scoreIndex2_s).x2, lineArray.at(scoreIndex2_s).y2), Scalar(0, 0, 255), 2, cv::LineTypes::LINE_4);

			cv::line(draw2, Point(bottomRow.x1, bottomRow.y1), Point(bottomRow.x2, bottomRow.y2), Scalar(0, 0, 255), 2, cv::LineTypes::LINE_4);
			cv::line(draw2, Point(topRow.x1, topRow.y1), Point(topRow.x2, topRow.y2), Scalar(255, 0, 0), 2, cv::LineTypes::LINE_4);

			rectangle2.push_back(bottomRow);
			rectangle2.push_back(topRow);
		}//if (scoreIndex2_o != -1)

		if (scoreIndex_o == -1 && scoreIndex2_o == -1)
		{
			return 0;
		}
		else
		{
			if (scoreIndex2_o != -1 && scoreIndex_o != scoreIndex2_o && scoreIndex_s != scoreIndex2_s)
				return 2;
			else return 1;
		}

		//cout << fixed << std::setprecision(2) << " Avg Length:" << average_Length << endl;
	}
}

inline void StereoImageBuffer::printHoughLines(std::vector<cv::Vec4i> allLines, Mat& img) const
{
	int totalNumOfLines = allLines.size();

	double previousDegree = -1000;

	if (!totalNumOfLines <= 0) //if there are lines/line
	{
		std::vector<double> degreeArray;
		std::vector<int> degreeArrayCount;

		double currentLine_m;
		double currentLineAngleRadian;
		double currentLineAngleDegree;
		double currentLineLength = 0;

		double average_Length = 0;

		int showLines = round(totalNumOfLines / 1);

		for (size_t i = 0; i < showLines; i++)
		{
			float x1 = allLines[i][0];
			float y1 = allLines[i][1];
			float x2 = allLines[i][2];
			float y2 = allLines[i][3];

			if (x2 != x1)
				currentLine_m = ((y2 - y1) / (x2 - x1));
			else
				currentLine_m = -1;

			currentLineAngleRadian = std::atan((y2 - y1) / (x2 - x1));
			currentLineAngleDegree = round(currentLineAngleRadian*(180 / CV_PI));
			currentLineLength = sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));
			//std::cout << fixed << std::setprecision(2) << i << ". Degree:" << currentLineAngleDegree << std::setprecision(0) << " --> Length: " << currentLineLength << " x1:" << x1 << " y1:" << y1 << " x2:" << x2 << " y2:" << y2 << endl;


			average_Length += currentLineLength;

			int dSize = degreeArray.size();

			if (dSize > 0)
			{
				bool found = false;
				for (size_t d = 0; d < dSize; d++)
				{
					double deg = degreeArray.at(d);
					if (deg == currentLineAngleDegree)
					{
						found = true;
						degreeArrayCount.at(d) = degreeArrayCount.at(d) + 1;
						break;
					}
				}

				if (!found)
				{
					degreeArray.push_back(currentLineAngleDegree);
					degreeArrayCount.push_back(1);
				}
			}
			else
			{
				degreeArray.push_back(currentLineAngleDegree);
				degreeArrayCount.push_back(1);
			}

			int r = rand() % 255;
			int g = rand() % 255;
			int b = rand() % 255;

			cv::line(img, Point(x1, y1), Point(x2, y2), Scalar(r, g, b), 1, cv::LineTypes::LINE_4);

			int mid = (y1 + y2) / 2;
			if (i % 2 == 0)
			{
				std::ostringstream text;
				text << i;// << ":" << fixed << std::setprecision(0) << currentLineAngleDegree << "(" << x1 << "," << y1 << ")";
				cv::putText(img, text.str(), Point(x1 + 10, mid), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(r, g, b), 1, CV_AA);
				text.str("");
				text.clear();
			}
			else
			{
				std::ostringstream text;
				text << i;// << ":" << fixed << std::setprecision(0) << currentLineAngleDegree << "(" << x1 << "," << y1 << ")";
				cv::putText(img, text.str(), Point(x1 - 10, mid), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(r, g, b), 1, CV_AA);
				text.str("");
				text.clear();
			}



		}

		//std::cout << endl << endl;
		//for (size_t deg = 0; deg < degreeArray.size(); deg++)
		//{
		//	std::cout << fixed << std::setprecision(2) << deg << ". Degree:" << degreeArray.at(deg) << " Count: " << degreeArrayCount.at(deg) << endl;
		//}

		//average_Length = average_Length / totalNumOfLines;
		//std::cout << fixed << std::setprecision(2) << " Avg Length:" << average_Length << endl;
	}
}

inline void StereoImageBuffer::getBoundingBoxes(Mat& src, Mat& disp, Mat& Dock, Mat& occupancyMatrix, Mat &line_draws) const
{
	/// Preparetion

	//Mat src_HLS;
	//cvtColor(src, src_HLS, COLOR_BGR2HLS);

	//Mat src_HSV;
	//cvtColor(src, src_HSV, COLOR_BGR2HSV);

	//// white color mask
	//Mat thresholded_WHITE;
	//Scalar lowerW(0, 200, 0); // np.uint8([0, 200, 0]);//r g b --> BGR
	//Scalar upperW(255, 255, 255);
	//inRange(src_HLS, lowerW, upperW, thresholded_WHITE);

	//// yellow color mask
	//Mat thresholded_YELLOW;
	//Scalar lowerY(10, 200, 100); // np.uint8([0, 200, 0]);//r g b
	//Scalar upperY(40, 255, 255);
	//inRange(src_HLS, lowerY, upperY, thresholded_YELLOW);

	/// Convert the image to grayscale
	Mat src_gray, src_blurred;
	if (src.channels() == 3)
	{
		cvtColor(src, src_gray, CV_BGR2GRAY);
	}
	else { src.copyTo(src_gray); }
	/// Convert the image to grayscale

	int w = round(src_gray.cols / 3);
	int h = round(src_gray.rows / 3);

	Mat src_32F;
	src_gray.convertTo(src_32F, CV_32FC1);

	/// Create windows
	//namedWindow("Edge Map", CV_WINDOW_NORMAL);
	//resizeWindow("Edge Map", w, h);
	//cv::moveWindow("Edge Map", 10, 50);

	////namedWindow("Good Matches", CV_WINDOW_NORMAL);
	////resizeWindow("Good Matches", w, h);
	////cv::moveWindow("Good Matches", w + 50, 50);

	//namedWindow("Merge Lines", CV_WINDOW_NORMAL);
	//resizeWindow("Merge Lines", w, h);
	//cv::moveWindow("Merge Lines", 10, h + 100);

	//namedWindow("Dock Shape", CV_WINDOW_NORMAL);
	//resizeWindow("Dock Shape", w, h);
	//cv::moveWindow("Dock Shape", w + 50, h + 100);
	/// Create windows

	int kernelSize_Common = 9;
	/// Preparetion

	// Find Edges
	Mat dst_edges, detected_edges;

	int edgeThresh = 1;
	int lowThreshold = 75;//160;175;
	int ratio = 2.0;
	int kernelSize_Blur = 9;
	int kernelSize_Edges = 3;

	/// Create a matrix of the same type and size as src (for dst)
	dst_edges.create(src.size(), src.type());

	/// Reduce noise with a kernel 3x9 (x9 since we want vertical lines)
	blur(src_gray, src_blurred, Size(3, kernelSize_Blur));

	/// Canny detector
	Canny(src_blurred, detected_edges, lowThreshold, lowThreshold*ratio, kernelSize_Edges);

	/// Using Canny's output as a mask, we display our result
	dst_edges = Scalar::all(0);

	//Mat detected_StrongEdges;
	//int kernelSize_StrengthenEdges = 3;
	//Mat kernel_Edges = getStructuringElement(cv::MorphShapes::MORPH_RECT, Size(kernelSize_StrengthenEdges, kernelSize_StrengthenEdges), Point(-1, -1));
	//dilate(detected_edges, detected_StrongEdges, kernel_Edges);

	//Find Vertical lines
	int houghLinesP_threshold = 30;
	double houghLinesP_minLineLength = 20;
	double houghLinesP_maxLineGap = 30;
	double houghLinesP_rho = 1;
	double houghLinesP_Degree = 1;
	double houghLinesP_theta = houghLinesP_Degree * (CV_PI / 180);

	std::vector<cv::Vec4i> allLines;
	HoughLinesP(detected_edges, allLines, houghLinesP_rho, houghLinesP_theta, houghLinesP_threshold, houghLinesP_minLineLength, houghLinesP_maxLineGap);

	Mat detected_houghlines, merged_houghlines;
	cvtColor(detected_edges, detected_houghlines, cv::COLOR_GRAY2BGR);
	cvtColor(detected_edges, merged_houghlines, cv::COLOR_GRAY2BGR);
	//printHoughLines(allLines, detected_houghlines);


	//src.copyTo(Dock);

	std::vector<MyLine> rectangle1, rectangle2;
	int numberOfRectangles = mergeLines(allLines, detected_houghlines, Dock, disp, rectangle1, rectangle2, occupancyMatrix, line_draws);
	//Find Vertical lines


	if (numberOfRectangles == 1)
	{
		cv::line(Dock, Point(rectangle1.at(0).realUp_x, rectangle1.at(0).realUp_y), Point(rectangle1.at(0).realBottom_x, rectangle1.at(0).realBottom_y), Scalar(0, 255, 0), 8, cv::LineTypes::LINE_4);
		cv::line(Dock, Point(rectangle1.at(1).realUp_x, rectangle1.at(1).realUp_y), Point(rectangle1.at(1).realBottom_x, rectangle1.at(1).realBottom_y), Scalar(0, 255, 0), 8, cv::LineTypes::LINE_4);
		cv::line(Dock, Point(rectangle1.at(2).x1, rectangle1.at(2).y1), Point(rectangle1.at(2).x2, rectangle1.at(2).y2), Scalar(0, 255, 0), 8, cv::LineTypes::LINE_4);
		cv::line(Dock, Point(rectangle1.at(3).x1, rectangle1.at(3).y1), Point(rectangle1.at(3).x2, rectangle1.at(3).y2), Scalar(0, 255, 0), 8, cv::LineTypes::LINE_4);
	}

	if (numberOfRectangles == 2)
	{
		cv::line(Dock, Point(rectangle1.at(0).realUp_x, rectangle1.at(0).realUp_y), Point(rectangle1.at(0).realBottom_x, rectangle1.at(0).realBottom_y), Scalar(0, 255, 0), 8, cv::LineTypes::LINE_4);
		cv::line(Dock, Point(rectangle1.at(1).realUp_x, rectangle1.at(1).realUp_y), Point(rectangle1.at(1).realBottom_x, rectangle1.at(1).realBottom_y), Scalar(0, 255, 0), 8, cv::LineTypes::LINE_4);
		cv::line(Dock, Point(rectangle1.at(2).x1, rectangle1.at(2).y1), Point(rectangle1.at(2).x2, rectangle1.at(2).y2), Scalar(0, 255, 0), 8, cv::LineTypes::LINE_4);
		cv::line(Dock, Point(rectangle1.at(3).x1, rectangle1.at(3).y1), Point(rectangle1.at(3).x2, rectangle1.at(3).y2), Scalar(0, 255, 0), 8, cv::LineTypes::LINE_4);


		cv::line(Dock, Point(rectangle2.at(0).realUp_x, rectangle2.at(0).realUp_y), Point(rectangle2.at(0).realBottom_x, rectangle2.at(0).realBottom_y), Scalar(0, 255, 0), 8, cv::LineTypes::LINE_4);
		cv::line(Dock, Point(rectangle2.at(1).realUp_x, rectangle2.at(1).realUp_y), Point(rectangle2.at(1).realBottom_x, rectangle2.at(1).realBottom_y), Scalar(0, 255, 0), 8, cv::LineTypes::LINE_4);
		cv::line(Dock, Point(rectangle2.at(2).x1, rectangle2.at(2).y1), Point(rectangle2.at(2).x2, rectangle2.at(2).y2), Scalar(0, 255, 0), 8, cv::LineTypes::LINE_4);
		cv::line(Dock, Point(rectangle2.at(3).x1, rectangle2.at(3).y1), Point(rectangle2.at(3).x2, rectangle2.at(3).y2), Scalar(0, 255, 0), 8, cv::LineTypes::LINE_4);
	}

	//src.copyTo(dst_edges, detected_edges);
	//imshow("Good Matches", SURF);
	//imshow("Edge Map", merged_houghlines);
	//imshow("Merge Lines", detected_houghlines);
	//imshow("Dock Shape", Dock);

	//waitKey(1000);


	////Find Corners
	Mat dst_corners, detected_corners8U, detected_corners;
	//int blockSize = 32; // - It is the size of neighbourhood considered for corner detection (see the details on cornerEigenValsAndVecs()).
	//int kernelSize_Corners = 5; // -Aperture parameter for the Sobel() operator.
	//int kernelSize_Strengthen = kernelSize_Common;
	//double k = 0.04; // -Harris detector free parameter in the equation.
	//dst_edges.create(src.size(), src.type());

	//cornerHarris(src_32F, detected_corners, blockSize, kernelSize_Corners, k, cv::BorderTypes::BORDER_DEFAULT);

	//double minVal, maxVal;
	//cv::minMaxLoc(detected_corners, &minVal, &maxVal);
	//detected_corners.convertTo(detected_corners8U, CV_8UC1, 255 / ((maxVal/4)-minVal));
	//
	////strengthen corners
	//Mat detected_StrongCorners, detected_StrongCornersBinary;
	//Mat kernel = getStructuringElement(cv::MorphShapes::MORPH_RECT, Size(kernelSize_Strengthen, kernelSize_Strengthen), Point(-1, -1));
	//dilate(detected_corners8U, detected_StrongCorners, kernel);	
	////strengthen corners
	//
	////Binary Corners
	//threshold(detected_StrongCorners, detected_StrongCornersBinary, 0, 255, cv::THRESH_OTSU);

	//dst_corners = Scalar::all(0);
	//src.copyTo(dst_corners, detected_StrongCornersBinary);

	//imshow("Corner Map", dst_corners);
	///// Create a Trackbar for user to enter threshold

	////Mat stats, centroids, centroids_32F;
	////connectedComponentsWithStats(output_Thresholded, output_Components, stats, centroids, 8, 4);

	////Mat detected_EdgesAndCorners;
	////detected_EdgesAndCorners.create(src.size(), src.type());

}

inline void StereoImageBuffer::ImageToWorldCoordinates(MyDockObject &obj)
{
	if (obj.d > 0)
	{
		//compute 3D point
		obj.Zs = (stereoParams.fx * stereoParams.b) / obj.d;
		obj.Xs = (obj.Zs / stereoParams.fx) * (obj.u - stereoParams.u0);
		//obj.Ys = (obj.Zs / fy) * (stereoParams.v0 - obj.v); //Upline of the object regarding to the camera position (zero (0) if parelel to the camera)
		obj.Ys = (obj.Zs / stereoParams.fy) * (stereoParams.v0 - obj.v);// just for check
		obj.distanceS = sqrt(pow(obj.Xs, 2) + pow(obj.Ys, 2) + pow(obj.Zs, 2));

		// correct with camera position and tilt angle
		obj.Xw = obj.Xs + stereoParams.cX; // (-b / 2)
		obj.Zw = obj.Zs *cos(stereoParams.tilt) + obj.Ys *sin(stereoParams.tilt) + stereoParams.cZ;
		obj.Yw = -obj.Zs*sin(stereoParams.tilt) + obj.Ys*cos(stereoParams.tilt) + stereoParams.cY;
		obj.distanceW = sqrt(pow(obj.Xw, 2) + pow(obj.Yw, 2) + pow(obj.Zw, 2));
	}
}
#endif

#ifdef WITH_BUMBLEBEE
inline  void StereoImageBuffer::setBumblebeeStereoParams(Bumblebee &obj)
{
	stereoParams.b = bumblebeeObject->stereoParams.b;
	stereoParams.cX = bumblebeeObject->stereoParams.cX;
	stereoParams.cY = bumblebeeObject->stereoParams.cY;
	stereoParams.cZ = bumblebeeObject->stereoParams.cZ;

	stereoParams.f = bumblebeeObject->stereoParams.f;

	stereoParams.sx = bumblebeeObject->stereoParams.sx;
	stereoParams.fx = bumblebeeObject->stereoParams.f / bumblebeeObject->stereoParams.sx;

	stereoParams.sy = bumblebeeObject->stereoParams.sy;
	stereoParams.fy = bumblebeeObject->stereoParams.f / bumblebeeObject->stereoParams.sy;

	stereoParams.tilt = bumblebeeObject->stereoParams.tilt;
	stereoParams.u0 = bumblebeeObject->stereoParams.u0;
	stereoParams.v0 = bumblebeeObject->stereoParams.v0;
}
#endif




