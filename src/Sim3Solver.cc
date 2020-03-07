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


#include "Sim3Solver.h"

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>

#include "KeyFrame.h"
#include "ORBmatcher.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

namespace ORB_SLAM2
{


Sim3Solver::Sim3Solver(KeyFrame *pKF1, KeyFrame *pKF2, const vector<MapPoint *> &vpMatched12, const bool bFixScale):
    mnIterations(0), mnBestInliers(0), mbFixScale(bFixScale)
{
    mpKF1 = pKF1;   //当前帧
    mpKF2 = pKF2;   //候选关键帧

    //取pKF1mappoint
    vector<MapPoint*> vpKeyFrameMP1 = pKF1->GetMapPointMatches();

    mN1 = vpMatched12.size();

    mvpMapPoints1.reserve(mN1);
    mvpMapPoints2.reserve(mN1);
    mvpMatches12 = vpMatched12;
    mvnIndices1.reserve(mN1);
    mvX3Dc1.reserve(mN1);
    mvX3Dc2.reserve(mN1);

    //取两帧的位姿
    cv::Mat Rcw1 = pKF1->GetRotation();
    cv::Mat tcw1 = pKF1->GetTranslation();
    cv::Mat Rcw2 = pKF2->GetRotation();
    cv::Mat tcw2 = pKF2->GetTranslation();

    mvAllIndices.reserve(mN1);

    size_t idx=0;
    // mN1为pKF1特征点的个数,遍历vpMatched12中匹配的每对的mappoint
    for(int i1=0; i1<mN1; i1++)
    {
        if(vpMatched12[i1])
        {
            // 得到两帧的mappoint集合
            // vpMatched12[当前帧pKF1某个特征点idx]=与pKF1第idx个特征点匹配的路标点(来自pKF2)
            // pMP1和pMP2是匹配的MapPoint
            MapPoint* pMP1 = vpKeyFrameMP1[i1];
            MapPoint* pMP2 = vpMatched12[i1];

            if(!pMP1)
                continue;

            if(pMP1->isBad() || pMP2->isBad())
                continue;

            //获取各自的特征点索引
            int indexKF1 = pMP1->GetIndexInKeyFrame(pKF1);
            int indexKF2 = pMP2->GetIndexInKeyFrame(pKF2);

            if(indexKF1<0 || indexKF2<0)
                continue;
            // indexKF1和indexKF2是匹配特征点的索引
            // 分别取mappoint对应的特征点
            const cv::KeyPoint &kp1 = pKF1->mvKeysUn[indexKF1];
            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[indexKF2];

            const float sigmaSquare1 = pKF1->mvLevelSigma2[kp1.octave];
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];

            mvnMaxError1.push_back(9.210*sigmaSquare1);
            mvnMaxError2.push_back(9.210*sigmaSquare2);

            //保存两帧的mappoint坐标
            mvpMapPoints1.push_back(pMP1);
            mvpMapPoints2.push_back(pMP2);
            mvnIndices1.push_back(i1);

            //保存当前帧pKF1的mappoint的转到pKF1相机坐标系的坐标
            //pMP1->GetWorldPos()：3x1的列向量
            cv::Mat X3D1w = pMP1->GetWorldPos();
            mvX3Dc1.push_back(Rcw1*X3D1w+tcw1);

            //保存当前帧pKF2的mappoint的转到pKF2相机坐标系的坐标
            cv::Mat X3D2w = pMP2->GetWorldPos();
            mvX3Dc2.push_back(Rcw2*X3D2w+tcw2);

            mvAllIndices.push_back(idx);
            idx++;
        }
    }
    //内参
    mK1 = pKF1->mK;
    mK2 = pKF2->mK;
    
    //相机模型？
    FromCameraToImage(mvX3Dc1,mvP1im1,mK1);
    FromCameraToImage(mvX3Dc2,mvP2im2,mK2);

    SetRansacParameters();
}

void Sim3Solver::SetRansacParameters(double probability, int minInliers, int maxIterations)
{
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;    

    N = mvpMapPoints1.size(); // number of correspondences

    mvbInliersi.resize(N);

    // Adjust Parameters according to number of correspondences
    float epsilon = (float)mRansacMinInliers/N;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;

    if(mRansacMinInliers==N)
        nIterations=1;
    else
        nIterations = ceil(log(1-mRansacProb)/log(1-pow(epsilon,3)));

    mRansacMaxIts = max(1,min(nIterations,mRansacMaxIts));

    mnIterations = 0;
}

cv::Mat Sim3Solver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
    //是否符合ransac的标准，也就是RANSAC是否成功
    bNoMore = false;
    vbInliers = vector<bool>(mN1,false);
    nInliers=0;

    //如果当前帧pKF1的mappoint数量比最低的成功要求还少，直接返回
    if(N<mRansacMinInliers)
    {
        bNoMore = true;
        return cv::Mat();
    }

    vector<size_t> vAvailableIndices;

    //随机抽取的3对点
    cv::Mat P3Dc1i(3,3,CV_32F);
    cv::Mat P3Dc2i(3,3,CV_32F);

    int nCurrentIterations = 0;
    while(mnIterations<mRansacMaxIts && nCurrentIterations<nIterations)
    {
        nCurrentIterations++;// 这个函数中迭代的次数
        mnIterations++;// 当前这个Sim3Solver总的迭代次数，默认为最大为300

        //为了随机抽取而准备的序列，如果匹配数为N，则mvAllIndices=[0,1,...,N-1]
        vAvailableIndices = mvAllIndices;

        // 步骤1：任意取三组点算Sim矩阵
        // Get min set of points
        // 从vAvailableIndices储存的序列集合中随机抽取3个数字
        for(short i = 0; i < 3; ++i)
        {
            //生成随机数
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);

            int idx = vAvailableIndices[randi];
            //将每个点(x_c,y_c,z_c)^T复制到列向量，一列代表一个点
            // P3Dc1i和P3Dc2i中点的排列顺序：
            // x1 x2 x3 ...
            // y1 y2 y3 ...
            // z1 z2 z3 ...
            mvX3Dc1[idx].copyTo(P3Dc1i.col(i));//mvpMapPoints1在相机mpKF1下的坐标
            mvX3Dc2[idx].copyTo(P3Dc2i.col(i));//mvpMapPoints2在相机mpKF2下的坐标

            //删除已经用过的随机数序号
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
	

        // 步骤2：根据3对匹配的3D点，计算之间的Sim3变换，也就是计算尺度s旋转R以及平移t
        ComputeSim3(P3Dc1i,P3Dc2i);

        // 步骤3：通过投影误差进行inlier检测
        CheckInliers();

        //取内点数最高的一种求解
        //更新mnBestInliers
        if(mnInliersi>=mnBestInliers)
        {
            mvbBestInliers = mvbInliersi;
            mnBestInliers = mnInliersi;
            mBestT12 = mT12i.clone();
            mBestRotation = mR12i.clone();
            mBestTranslation = mt12i.clone();
            mBestScale = ms12i;

            // 只要计算得到一次合格的Sim变换，就直接返回
            if(mnInliersi>mRansacMinInliers)
            {
                nInliers = mnInliersi;
                for(int i=0; i<N; i++)
                    if(mvbInliersi[i])
                        vbInliers[mvnIndices1[i]] = true;
                return mBestT12;
            }
        }
    }

    //总的迭代次数超过阈值mRansacMaxIts都还有没达到mnInliersi>mRansacMinInliers的要求，于是bNoMore=true
    if(mnIterations>=mRansacMaxIts)
        bNoMore=true;

    return cv::Mat();
}

cv::Mat Sim3Solver::find(vector<bool> &vbInliers12, int &nInliers)
{
    bool bFlag;
    return iterate(mRansacMaxIts,bFlag,vbInliers12,nInliers);
}

void Sim3Solver::ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C)
{
    //Reduces a matrix to a vector
    //CV_REDUCE_SUM-输出是矩阵的所有行/列的和.
    //第3个参数为1： 表示将输入的3个点变成一个列向量，即对矩阵每一行求和，变成3x1列向量
    cv::reduce(P,C,1,CV_REDUCE_SUM);
    //求均值
    //得到3个点的均值(x_bar,y_bar,z_bar)^T
    C = C/P.cols;

    //变成去质心坐标
    for(int i=0; i<P.cols; i++)
    {
        Pr.col(i)=P.col(i)-C;
    }
}

void Sim3Solver::ComputeSim3(cv::Mat &P1, cv::Mat &P2)
{
    // ！！！！！！！这段代码一定要看这篇论文！！！！！！！！！！！
    // Custom implementation of:
    // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

    // Step 1: Centroid and relative coordinates

    // Pr1: 3x3矩阵，每一列是一个点
    cv::Mat Pr1(P1.size(),P1.type()); // Relative coordinates to centroid (set 1)
    cv::Mat Pr2(P2.size(),P2.type()); // Relative coordinates to centroid (set 2)
    cv::Mat O1(3,1,Pr1.type()); // Centroid of P1
    cv::Mat O2(3,1,Pr2.type()); // Centroid of P2

    //Pr1,Pr2为去质心坐标，O1为质心
    ComputeCentroid(P1,Pr1,O1);
    ComputeCentroid(P2,Pr2,O2);

    /// 接下来并没有选择直接求解旋转矩阵，
    /// 为了使后面求尺度因子时的误差最小化，这里先采用四元数解法来求R，具体看论文推导

    // Step 2: Compute M matrix
    // M: 去质心点矩阵相乘
    cv::Mat M = Pr2*Pr1.t();

    // Step 3: Compute N matrix

    double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

    cv::Mat N(4,4,P1.type());

    N11 = M.at<float>(0,0)+M.at<float>(1,1)+M.at<float>(2,2);
    N12 = M.at<float>(1,2)-M.at<float>(2,1);
    N13 = M.at<float>(2,0)-M.at<float>(0,2);
    N14 = M.at<float>(0,1)-M.at<float>(1,0);
    N22 = M.at<float>(0,0)-M.at<float>(1,1)-M.at<float>(2,2);
    N23 = M.at<float>(0,1)+M.at<float>(1,0);
    N24 = M.at<float>(2,0)+M.at<float>(0,2);
    N33 = -M.at<float>(0,0)+M.at<float>(1,1)-M.at<float>(2,2);
    N34 = M.at<float>(1,2)+M.at<float>(2,1);
    N44 = -M.at<float>(0,0)-M.at<float>(1,1)+M.at<float>(2,2);

    N = (cv::Mat_<float>(4,4) << N11, N12, N13, N14,
                                 N12, N22, N23, N24,
                                 N13, N23, N33, N34,
                                 N14, N24, N34, N44);


    // Step 4: Eigenvector of the highest eigenvalue

    cv::Mat eval, evec;
    //对矩阵N进行特征值分解
    cv::eigen(N,eval,evec); //evec[0] is the quaternion of the desired rotation

    //取最大特征值对应的特征向量，是与旋转矩阵有关的四元数
    cv::Mat vec(1,3,evec.type());
    (evec.row(0).colRange(1,4)).copyTo(vec); //extract imaginary part of the quaternion (sin*axis)

    // Rotation angle. sin is the norm of the imaginary part, cos is the real part
    // 从四元数提取旋转的角度
    // 四元数 q= cos(\theta * 0.5)+ n_{3x1} * sin(\theta *0.5)
    double ang=atan2(norm(vec),evec.at<float>(0,0));

    // 将四元素变成轴角表示， 乘以2是因为从四元数提取出来的角度只是一半
    vec = 2*ang*vec/norm(vec); //Angle-axis representation. quaternion angle is the half

    //罗德里格斯，将旋转向量转为旋转矩阵，保存到mR12i
    // 注意：这里求出来的旋转是 从相机坐标系2到相机坐标系1的变换，
    mR12i.create(3,3,P1.type());
    cv::Rodrigues(vec,mR12i); // computes the rotation matrix from angle-axis

    // Step 5: Rotate set 2

    //将去质心坐标Pr2转换到第一帧坐标系下(这里只是旋转对齐，还没有加平移)
    cv::Mat P3 = mR12i*Pr2;

    // Step 6: Scale
    // 先检查是不是双目或者RGBD这些固定尺度的情况
    if(!mbFixScale)
    {
        //不是，则求解尺度因子,按照论文公式来
        //两个矩阵求内积？
        double nom = Pr1.dot(P3);
        //aux_P3: 3x3
        cv::Mat aux_P3(P3.size(),P3.type());
        aux_P3=P3;
        cv::pow(P3,2,aux_P3);   //矩阵元素各自平方
        double den = 0;

        for(int i=0; i<aux_P3.rows; i++)
        {
            for(int j=0; j<aux_P3.cols; j++)
            {
                den+=aux_P3.at<float>(i,j);
            }
        }
        //两个点云的尺度
        ms12i = nom/den;
    }
    else
        ms12i = 1.0f;

    // Step 7: Translation
    // 计算平移
    mt12i.create(1,3,P1.type());
    mt12i = O1 - ms12i*mR12i*O2;

    // Step 8: Transformation

    // Step 8.1 T12
    // 组合成变换矩阵T
    mT12i = cv::Mat::eye(4,4,P1.type());

    // sR_12
    cv::Mat sR = ms12i*mR12i;

    sR.copyTo(mT12i.rowRange(0,3).colRange(0,3));
    mt12i.copyTo(mT12i.rowRange(0,3).col(3));

    // Step 8.2 T21

    mT21i = cv::Mat::eye(4,4,P1.type());

    // s'*R_21=1/(s*R_12)=(1/s)*(R_12^T)
    cv::Mat sRinv = (1.0/ms12i)*mR12i.t();

    sRinv.copyTo(mT21i.rowRange(0,3).colRange(0,3));
    cv::Mat tinv = -sRinv*mt12i;
    tinv.copyTo(mT21i.rowRange(0,3).col(3));
}


void Sim3Solver::CheckInliers()
{
    vector<cv::Mat> vP1im2, vP2im1;
    ////将mvX3Dc2中的3d点通过参数mT12i，mK1投影为2d像素坐标，放入vP2im1
    Project(mvX3Dc2,vP2im1,mT12i,mK1);  // 把2系中的3D经过Sim3变换(mT12i)到1系中计算重投影坐标
    Project(mvX3Dc1,vP1im2,mT21i,mK2);  // 把1系中的3D经过Sim3变换(mT21i)到2系中计算重投影坐标

    mnInliersi=0;

    //判定mvP1im1中的点哪些是内点
    for(size_t i=0; i<mvP1im1.size(); i++)
    {
        cv::Mat dist1 = mvP1im1[i]-vP2im1[i];
        cv::Mat dist2 = vP1im2[i]-mvP2im2[i];

        const float err1 = dist1.dot(dist1);
        const float err2 = dist2.dot(dist2);

        if(err1<mvnMaxError1[i] && err2<mvnMaxError2[i])
        {
            mvbInliersi[i]=true;
            mnInliersi++;
        }
        else
            mvbInliersi[i]=false;
    }
}


cv::Mat Sim3Solver::GetEstimatedRotation()
{
    return mBestRotation.clone();
}

cv::Mat Sim3Solver::GetEstimatedTranslation()
{
    return mBestTranslation.clone();
}

float Sim3Solver::GetEstimatedScale()
{
    return mBestScale;
}

void Sim3Solver::Project(const vector<cv::Mat> &vP3Dw, vector<cv::Mat> &vP2D, cv::Mat Tcw, cv::Mat K)
{
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    const float &fx = K.at<float>(0,0);
    const float &fy = K.at<float>(1,1);
    const float &cx = K.at<float>(0,2);
    const float &cy = K.at<float>(1,2);

    vP2D.clear();
    vP2D.reserve(vP3Dw.size());

    for(size_t i=0, iend=vP3Dw.size(); i<iend; i++)
    {
        cv::Mat P3Dc = Rcw*vP3Dw[i]+tcw;
        const float invz = 1/(P3Dc.at<float>(2));
        const float x = P3Dc.at<float>(0)*invz;
        const float y = P3Dc.at<float>(1)*invz;

        vP2D.push_back((cv::Mat_<float>(2,1) << fx*x+cx, fy*y+cy));
    }
}

void Sim3Solver::FromCameraToImage(const vector<cv::Mat> &vP3Dc, vector<cv::Mat> &vP2D, cv::Mat K)
{
    const float &fx = K.at<float>(0,0);
    const float &fy = K.at<float>(1,1);
    const float &cx = K.at<float>(0,2);
    const float &cy = K.at<float>(1,2);

    vP2D.clear();
    vP2D.reserve(vP3Dc.size());

    for(size_t i=0, iend=vP3Dc.size(); i<iend; i++)
    {
        const float invz = 1/(vP3Dc[i].at<float>(2));
        const float x = vP3Dc[i].at<float>(0)*invz;
        const float y = vP3Dc[i].at<float>(1)*invz;

        vP2D.push_back((cv::Mat_<float>(2,1) << fx*x+cx, fy*y+cy));
    }
}

} //namespace ORB_SLAM
