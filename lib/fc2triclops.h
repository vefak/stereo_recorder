  //=============================================================================
  // Copyright © 2016 FLIR Integrated Imaging Solutions, Inc. All Rights Reserved.
  //
  // This software is the confidential and proprietary information of FLIR
  // Integrated Imaging Solutions, Inc. ("Confidential Information"). You
  // shall not disclose such Confidential Information and shall use it only in
  // accordance with the terms of the license agreement you entered into
  // with FLIR Integrated Imaging Solutions, Inc. (FLIR).
  //
  // FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
  // SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
  // IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
  // PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
  // SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
  // THIS SOFTWARE OR ITS DERIVATIVES.
  //=============================================================================
  
  //=============================================================================
  // $Id: fc2triclops.h 309688 2016-12-21 18:28:11Z silvanoa $
  //=============================================================================
  
  //=============================================================================
  //=============================================================================
  
  #ifndef FC2TRICLOPS_H
  #define FC2TRICLOPS_H
  
  //=============================================================================
  // PGR Includes
  //=============================================================================
  
  #include "flycapture2bridge.h"
  #include <FlyCapture2.h>
  
  //=============================================================================
  // System Includes
  //=============================================================================
  
  #include <assert.h>
  
  namespace Fc2Triclops {
  
  enum StereoCameraMode {
      TWO_CAMERA = 1,       
      TWO_CAMERA_NARROW,    
      TWO_CAMERA_WIDE,      
  };
  
  ErrorType
  setStereoMode(FlyCapture2::Camera &cam, StereoCameraMode &mode);
  
  ErrorType
  isBB2(const FlyCapture2::Camera &cam, bool &flag);
  
  ErrorType
  isBBXB3(const FlyCapture2::Camera &cam, bool &flag);
  
  int
  handleFc2Error(FlyCapture2::Error const &fc2Error);
  
  int
  handleFc2TriclopsError(Fc2Triclops::ErrorType const &error,
                         const char *pCallNameStr);
   // End of group FlyCapture2Bridge
  
  } // end namespace Fc2Triclops
  
  #endif // FC2TRICLOPS_H