/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{
//静态变量初始化
long unsigned int Frame::nNextId=0;
//标志此帧是否是初始化帧
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}

// 双目的初始化
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    //另开两个线程提取左右两张图片的特征点
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    //在这里等待两个线程介绍再往下进行
    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    //关键点畸变矫正
    UndistortKeyPoints();

    //计算匹配，同时获取深度
    ComputeStereoMatches();

    // 对应的mappoints
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

// RGBD初始化
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    //关键点畸变矫正
    UndistortKeyPoints();

    //计算特征点深度
    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    //储存哪些关键点是离群值
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;
    
    //将特征点分配到窗格中以加速特征点匹配
    AssignFeaturesToGrid();
}

// 单目初始化
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();     //从特征提取器获取高斯金字塔层数
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();//每层之间的缩放比例
    mfLogScaleFactor = log(mfScaleFactor);  //取对数
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();//每层的相对于原始图像的缩放比例,其值单调递增
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();//mvScaleFactor的倒数
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();//mvScaleFactor的平方
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();//mvScaleFactor的平方的倒数

    // ORB extraction
    // 提取orb特征点
    ExtractORB(0,imGray);

    N = mvKeys.size();//特征点数量

    if(mvKeys.empty())
        return;

    //关键点畸变纠正
    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    // 只在第一帧进行,或者更改标定参数之后
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);


        //FRAME_GRID_COLS:窗格列数
        //FRAME_GRID_ROWS:窗格行数
        //mnMaxX-mnMinX: 去畸变之后图像的列数
        //mnMaxY-mnMinY: 去畸变之后图像的高
        //目的: 将去畸变后的图像划分为 FRAME_GRID_COLS * FRAME_GRID_ROWS 个区域
        //      然后将特征点分配到这些区域
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    //将特征点分配到窗格中以加速特征点匹配
    AssignFeaturesToGrid();
}

//将特征点分配到窗格中以加速特征点匹配
void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    // mGrid分配空间
    // mGrid:储存这各个窗格的特征点在mvKeysUn中的序号
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i]; //mvKeysUn:纠正后的关键点

        int nGridPosX, nGridPosY;   //窗格索引
        if(PosInGrid(kp,nGridPosX,nGridPosY))   //获取当前关键点在去畸变图像后的对应窗格的索引
            mGrid[nGridPosX][nGridPosY].push_back(i);//储存当前帧各个窗格的特征点在mvKeysUn中的序号
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

/**
 * @brief Set the camera pose.
 *
 * 设置相机姿态，随后会调用 UpdatePoseMatrices() 来改变mRcw,mRwc等变量的值
 * @param Tcw Transformation from world to camera
 */
void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}


void Frame::UpdatePoseMatrices()
{ 
    // [x_camera 1] = [R|t]*[x_world 1]，坐标为齐次形式
    // x_camera = R*x_world + t
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    // mtcw, 即相机坐标系下相机坐标系到世界坐标系间的向量, 向量方向由相机坐标系指向世界坐标系
    // mOw, 即世界坐标系下世界坐标系到相机坐标系间的向量, 向量方向由世界坐标系指向相机坐标系
    mOw = -mRcw.t()*mtcw;
}

/**
 * @brief 判断一个点是否在视野内
 *
 * 计算了重投影坐标，观测方向夹角，预测在当前帧的尺度
 * @param  pMP             MapPoint
 * @param  viewingCosLimit 视角和平均视角的方向阈值
 * @return                 true if is in view
 * @see SearchLocalPoints()
 */
bool Frame:: isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    // 将这个点转换到相机坐标系
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    //如果z值为负，舍弃
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    // 相机模型投影到图像,得到图像点(u,v)
    // 这个(u,v)在后面用来做特征点匹配搜索, 采用范围搜索
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    //
    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    //检查观测角是否在阈值以内
    cv::Mat Pn = pMP->GetNormal();
    //观测角的cos值
    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    // 在局部局部地图跟踪TrackLocalMap()-->	SearchLocalPoints()-->matcher.SearchByProjection()用到
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz; //该3D点投影到双目右侧相机上的横坐标
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}
/**
 * @brief 找到在 以x,y为中心,边长为2r的方形内且在[minLevel, maxLevel]的特征点
 * @param x        图像坐标u
 * @param y        图像坐标v
 * @param r        边长
 * @param minLevel 最小尺度
 * @param maxLevel 最大尺度
 * @return         满足条件的特征点的序号
 */
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);
    
    //接下来计算方形的四边在哪在mGrid中的行数和列数
    //nMinCellX是方形左边在mGrid中的列数，如果它比mGrid的列数大，说明方形内肯定没有特征点，于是返回
    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }
		//确认此特征点在方形内
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;
		
		
                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{

    //posX= 64* (pt.x-mnMinX)/(mnMaxX-mnMinX) = 64 * 关键点x在去畸变后的图像上横坐标/去畸变图像的宽 = 关键点在去畸变图像上的哪个窗格
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    // 判断去除畸变之后的关键点是否在
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}
// 调用OpenCV的矫正函数矫正orb提取的特征点
void Frame::UndistortKeyPoints()
{
    // 如果没有图像是矫正过的，没有失真
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    //储存orb坐标
    // N为提取的特征点数量，将N个特征点保存在N*2的mat中
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    // 调整mat的通道为2，矩阵的行列形状不变
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    // 存储校正后的特征点
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0) //判断是否有畸变参数
    {
        // 矫正前四个边界点：(0,0) (cols,0) (0,rows) (cols,rows)
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));    //左上和左下横坐标最小的
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));    //右上和右下横坐标最大的
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));    //左上和右上纵坐标最小的
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));    //左下和右下纵坐标最小的

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

/**
 * @brief 双目匹配
 *
 * 为左图的每一个特征点在右图中找到匹配点 \n
 * 根据基线(有冗余范围)上描述子距离找到匹配, 再进行SAD精确定位 \n
 * 最后对所有SAD的值进行排序, 剔除SAD值较大的匹配对，然后利用抛物线拟合得到亚像素精度的匹配 \n
 * 匹配成功后会更新 mvuRight(ur) 和 mvDepth(Z)
 */
void Frame::ComputeStereoMatches()
{
    //初始化
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    // ORB阈值
    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    // Assign keypoints to row table
    // 步骤1：建立特征点搜索范围对应表，一个特征点在一个带状区域内搜索匹配特征点
    // 匹配搜索的时候，不仅仅是在一条横线上搜索，而是在一条横向搜索带上搜索,简而言之，原本每个特征点的纵坐标为1，这里把特征点体积放大，纵坐标占好几行
    // 例如左目图像某个特征点的纵坐标为20，那么在右侧图像上搜索时是在纵坐标为18到22这条带上搜索，搜索带宽度为正负2，搜索带的宽度和特征点所在金字塔层数有关
    // 简单来说，如果纵坐标是20，特征点在左图像第20行，那么认为右图18 19 20 21 22行都有这个特征点
    // vRowIndices[18]、vRowIndices[19]、vRowIndices[20]、vRowIndices[21]、vRowIndices[22]都有这个特征点编号
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    //右目特征点数
    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        //遍历右目特征点
        // !!在这个函数中没有对双目进行校正，双目校正是在外层程序中实现的
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        // 计算匹配搜索的纵向宽度，尺度越大（层数越高，距离越近），搜索范围越大
        // 如果特征点在金字塔第一层，则搜索范围为:正负2
        // 尺度越大其位置不确定性越高，所以其搜索半径越大
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        //根据极线
        //认为左目图像中(与右目某个特征点所在行及其r范围内的行)范围内的特征点与右目的这个特征点是匹配的
        //即左目右目特征点按行匹配
        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;           // 最小视差, 设置为0即可
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    // 对于左目图片的每一个特征点，在右目图片上寻找匹配
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    // 步骤2：对左目相机每个特征点，通过描述子在右目带状搜索区域找到匹配点, 再通过SAD做亚像素匹配
    // 注意：这里是校正前的mvKeys，而不是校正后的mvKeysUn
    // KeyFrame::UnprojectStereo和Frame::UnprojectStereo函数中不一致
    for(int iL=0; iL<N; iL++)   //遍历左目特征点
    {
        // 取一个特征点
        const cv::KeyPoint &kpL = mvKeys[iL];
        // 取特征点的层数
        const int &levelL = kpL.octave;
        // 取(u,v)
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        //取以(左目特征点所在行)的右目特征点候选列表
        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        // 步骤2.1：遍历右目所有可能的匹配点，找出最佳匹配点（描述子距离最小）
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            //遍历候选， 这都是右目上的特征点
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                // ORB描述符距离计算
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                //取最优匹配
                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        // 子像素匹配
        // 步骤2.2：通过SAD匹配提高像素匹配修正量bestincR
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            // kpL.pt.x对应金字塔最底层坐标，将最佳匹配的特征点对尺度变换到尺度对应层 (scaleduL, scaledvL) (scaleduR0, )
            const float uR0 = mvKeysRight[bestIdxR].pt.x;               //右目最佳匹配单的x值
            const float scaleFactor = mvInvScaleFactors[kpL.octave];    //左目特征点金字塔层数
            const float scaleduL = round(kpL.pt.x*scaleFactor);         //尺度调整到对应的层
            const float scaledvL = round(kpL.pt.y*scaleFactor);         //
            const float scaleduR0 = round(uR0*scaleFactor);             //右目特征点x值也乘以左目这个特征点的尺度因子

            // sliding window search
            // 滑动窗口搜索
            ///SAD(Sum of absolute differences)是一种图像匹配算法。
            ///基本思想：差的绝对值之和。
            ///此算法常用于图像块匹配，将每个像素对应数值之差的绝对值(L1)求和，据此评估两个图像块的相似度。
            ///该算法快速、但并不精确，通常用于多级处理的初步筛选
            const int w = 5;    // 滑动窗口的大小11*11 注意该窗口取自resize后的图像
            // 取左目的图像块
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F); //窗口中的每个元素减去正中心的那个元素，简单归一化，减小光照强度影响

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            // 滑动窗口的滑动范围为（-L, L）,提前判断滑动窗口滑动过程中是否会越界
            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                // 横向滑动窗口
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F); //窗口中的每个元素减去正中心的那个元素，简单归一化，减小光照强度影响

                // 计算两个图像块之间的距离
                float dist = cv::norm(IL,IR,cv::NORM_L1);// 一范数，计算差的绝对值
                // 取最优匹配的图像块
                if(dist<bestDist)
                {
                    bestDist =  dist;   // SAD匹配目前最小匹配偏差
                    bestincR = incR;    // SAD匹配目前最佳的修正量
                }
                // 储存图像块之间的距离
                vDists[L+incR] = dist; // 正常情况下，这里面的数据应该以抛物线形式变化
            }

            // 整个滑动窗口过程中，SAD最小值不是以抛物线形式出现，SAD匹配失败，同时放弃求该特征点的深度
            // 就是说，如果最优匹配出现在边界的地方，则认为不对，跳过
            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            // Sub-pixel match (Parabola fitting)
            // 步骤2.3：做抛物线拟合找谷底得到亚像素匹配deltaR
            // (bestincR,dist) (bestincR-1,dist) (bestincR+1,dist)三个点拟合出抛物线
            // bestincR+deltaR就是抛物线谷底的位置，相对SAD匹配出的最小值bestincR的修正量为deltaR
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];     // 取连续的3相邻窗口的的匹配距离
            const float dist3 = vDists[L+bestincR+1];

            // 拟合出二次曲线，取最低点对应的x值与dist2所对应的x值的差
            // bestincR: dist2对应的x值
            // 即有(x-1,dist1),(x,dist2),(x+1,dist3)3个点
            // 可使用 ax^2+bx+c=0来表达这个二次曲线，将上述3点代入
            // 消元，消去变量c，得到a，b表达式，就可以得到极值的x值 = -b/(2a)
            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            // 抛物线拟合得到的修正量不能超过一个像素，否则放弃求该特征点的深度
            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            // 通过描述子匹配得到匹配点位置为scaleduR0
            // 通过SAD匹配找到修正量bestincR
            // 通过抛物线拟合找到亚像素修正量deltaR
            // 经过调整之后的的右目最佳匹配点的x值
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            // 求视差
            float disparity = (uL-bestuR);

            // 视差阈值限定
            if(disparity>=minD && disparity<maxD)
            {
                //视差为负数
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                // 计算左目特征点深度 z=f*b/disparity
                mvDepth[iL]=mbf/disparity;
                // 储存左目特征点对应的右目特征点的x值
                mvuRight[iL] = bestuR;
                // 储存匹配对
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    // 步骤3：剔除SAD匹配偏差较大的匹配特征点
    // 前面SAD匹配只判断滑动窗口中是否有局部最小值，这里通过对比剔除SAD匹配偏差比较大的特征点的深度

    // 匹配对排序
    sort(vDistIdx.begin(),vDistIdx.end());   // 根据所有匹配对的SAD偏差进行排序, 距离由小到大
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;   // 计算自适应距离, 大于此距离的匹配对将剔除

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    // mvDepth直接由depth图像读取
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

         // mvDepth直接由depth图像读取
        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            //设置深度z
            mvDepth[i] = d;
            //根据深度z反求 虚拟右目的匹配的右目特征点的x值
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}
/**
 * @brief Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
 * @param  i 第i个keypoint
 * @return   3D点（相对于世界坐标系）
 */
//创建索引为i的关键点对应的世界坐标点(路标点)
cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i]; //取对应深度
    if(z>0)
    {
        //关键点反投影为相机坐标系下的点
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        //相机坐标系转换到世界坐标系
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

} //namespace ORB_SLAM
