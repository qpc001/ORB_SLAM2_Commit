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

#include "Initializer.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include "Optimizer.h"
#include "ORBmatcher.h"

#include<thread>

namespace ORB_SLAM2
{
Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
{
    /// 使用参考帧初始化初始化器

    // 拷贝内参
    mK = ReferenceFrame.mK.clone();

    // 拷贝参考帧的特征点
    mvKeys1 = ReferenceFrame.mvKeysUn;

    mSigma = sigma;
    mSigma2 = sigma*sigma;
    mMaxIterations = iterations;
}

bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                             vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
{
    // Fill structures with current keypoints and matches with reference frame
    // Reference Frame: 1, Current Frame: 2
    // 将当前帧特征点填入mvKeys2
    mvKeys2 = CurrentFrame.mvKeysUn;
    
    //mvMatches12储存着匹配点对在参考帧F1和当前帧F2中的序号
    mvMatches12.clear();
    mvMatches12.reserve(mvKeys2.size());
    //描述参考帧F1中特征点匹配情况
    mvbMatched1.resize(mvKeys1.size());
    //vMatches12: 初始化帧(参考帧)的特征点索引向量,其值是特征点序号
    //vMatches12[参考帧特征点idx]=当前帧特征点idx
    for(size_t i=0, iend=vMatches12.size();i<iend; i++)
    {
        if(vMatches12[i]>=0)
        {
            //mvMatches12[i]=pair(参考帧特征点idx,当前帧特征点idx)
            //mvbMatched1[i]= (bool)描述当前帧有没有特征点与参考帧是匹配的
            mvMatches12.push_back(make_pair(i,vMatches12[i]));
            mvbMatched1[i]=true;
        }
        else
            mvbMatched1[i]=false;
    }
    
    //匹配点数
    const int N = mvMatches12.size();

    // Indices for minimum set selection
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    //vAllIndices[1]=1 , vAllIndices[2]=2 , vAllIndices[3]=3 ...
    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    // Generate sets of 8 points for each RANSAC iteration
    // 在所有匹配特征点对中随机选择8对匹配特征点为一组，共选择mMaxIterations组
    // 用于FindHomography和FindFundamental求解
    // mMaxIterations:200
    mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));  //初始化,分配空间这个vector<vector> 是 200 * 8 的

    DUtils::Random::SeedRandOnce(0);

    //RANSAC循环mMaxIterations次
    for(int it=0; it<mMaxIterations; it++)
    {
        //每一次重新取8对点之前, 重新初始化vAvailableIndices
        //vAvailableIndices[1]=1 , vAvailableIndices[2]=2 , vAvailableIndices[3]=3 ...
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        // 选择一个最小集合, 即随机抽取8对匹配特征点
        for(size_t j=0; j<8; j++)
        {
            //RandomInt():返回一个随机int
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            int idx = vAvailableIndices[randi]; //idx=randi

            //储存这个 8对特征点集合
            mvSets[it][j] = idx;

            //用最后一个值往上面被抽中的位置覆盖idx
            vAvailableIndices[randi] = vAvailableIndices.back();
            //然后把最后一个值去掉,否则就重复了
            vAvailableIndices.pop_back();
        }
    }

    // Launch threads to compute in parallel a fundamental matrix and a homography
    // vector<bool> vbMatchesInliersH储存: 哪些匹配点对能够通过H重投影成功
    vector<bool> vbMatchesInliersH, vbMatchesInliersF;
    //SH计算单应矩阵的得分，SF计算基础矩阵得分
    float SH, SF;
    cv::Mat H, F;

    //启动线程，第一个参数是启用函数的指针，后面是调用这个函数所需的参数
    //由于FindHomography第二三个参数是引用，所以需要用ref()包裹,线程引用传值,如果用&则不能通过编译
    // ref():https://zhidao.baidu.com/question/1240776856100751219.html
    // 计算homograpy和得分
    thread threadH(&Initializer::FindHomography,this,ref(vbMatchesInliersH), ref(SH), ref(H));
    // 计算fundamental和得分
    thread threadF(&Initializer::FindFundamental,this,ref(vbMatchesInliersF), ref(SF), ref(F));

    // Wait until both threads have finished
    //在这里等待线程threadH，threadF结束才往下继续执行
    //也就是等待SH，SF的结果
    threadH.join();
    threadF.join();

    // Compute ratio of scores
    // 计算分数的比值
    float RH = SH/(SH+SF);

    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    // 从H矩阵或F矩阵中恢复R,t
    // 根据分数的比值来选择从H恢复还是从F恢复R,t
    if(RH>0.40)
        return ReconstructH(vbMatchesInliersH,H,mK,R21,t21,vP3D,vbTriangulated,1.0,50);
    else //if(pF_HF>0.6)
        return ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50);

    return false;
}


void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
{
    // Number of putative matches
    //假定匹配的数量
    const int N = mvMatches12.size();

    // Normalize coordinates
    // 坐标归一化
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2inv = T2.inv();   //求出T2^{-1}

    // Best Results variables
    // 这是要输出的,描述某个特征点是否内点
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat H21i, H12i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
        // RANSAC迭代200次,取得分最高情况下算出来的单应矩阵H
        // Select a minimum set 每个集合8对点(8点法)
        // mvSets[当前迭代次数][0~7]= 某个最小集合随机索引idx
        for(size_t j=0; j<8; j++)
        {
            int idx = mvSets[it][j];    //idx用来取某一堆匹配点对
            //mvMatches12[i]=pair(参考帧特征点idx,当前帧特征点idx), 这是匹配的特征点
            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        //计算H
        cv::Mat Hn = ComputeH21(vPn1i,vPn2i);
        H21i = T2inv*Hn*T1; //注意: 这里是T2的逆,不是转置, 原因是约束方程里是叉乘关系, u2 X H * u1 = 0
        H12i = H21i.inv();

        //在参数 mSigma下，能够通过H21，H12重投影成功的点有哪些，并返回分数
        currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);

        //只取最高得分情况下的H以及内点判断vbMatchesInliers
        if(currentScore>score)
        {
            H21 = H21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}


void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    // Number of putative matches
    const int N = vbMatchesInliers.size();

    // Normalize coordinates
    // 归一化点
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2t = T2.t();

    // Best Results variables
    // 输出
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(int j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            //mvMatches12[i]=pair(参考帧特征点idx,当前帧特征点idx)
            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        //计算出归一化特征点对应的基础矩阵
        //由基础矩阵约束 u2'*F*u1=0
        //可以得到,使用归一化的点代入,有:  _u2' * _F * _u1 =0
        //下面求出来的Fn就是 上式子的 _F
        cv::Mat Fn = ComputeF21(vPn1i,vPn2i);

        //又根据 _u2=T2*u2 , _u1=T1*u1
        //可得: _u2' * _F * _u1 = (T2*u2)' * _F *(T1*u1) = u2'*(T2' * _F * T1)* u1 = 0
        //即  :u2'*(T2' * _F * T1)* u1 = u2' * F * u1=0
        //所以: 当我们求解[u2'*F*u1=0]中的F的时候, 实际上这个F应该等于(T2' * _F * T1)

        //转换成归一化前特征点对应的基础矩阵
        //也就是求出没有归一化的点代入方程[u2'*F*u1=0]得到的F
        F21i = T2t*Fn*T1;

        //在参数 mSigma下，能够通过F21li，
        //重投影成功的点有哪些，并返回分数
        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

        //储存最高分的情况下的基础矩阵F
        if(currentScore>score)
        {
            F21 = F21i.clone(); //F
            vbMatchesInliers = vbCurrentInliers;    //内点判断(std::vector)(判断哪些特征点对是内点)
            score = currentScore;
        }
    }
}

//SVD分解求H21矩阵
cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(2*N,9,CV_32F);

    //虽然书上说最少用4个点对就可以解出单应矩阵，但是这里依然用的是8个点对
    for(int i=0; i<N; i++)
    {
        //构造A矩阵
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;

    }

    cv::Mat u,w,vt;

    //SVD分解A=u*w*vt
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    //返回了vt最后一个行向量 9维的向量
    //reshape成3x3
    return vt.row(8).reshape(0, 3);
}

//8点法求基础矩阵F,并根据约束进行调整F
cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(N,9,CV_32F);

    //八点法计算F

    //经过线性变换 DLT
    //构造矩阵A=u2^T F
    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    //用SVD算出基础矩阵
    //vt [8x8] : 由特征向量组成的矩阵, 每个特征向量都是9维的
    //vt.row(8):矩阵A经过SVD分解之后最小奇异值对应的特征向量9 维
    //重新reshape成3x3矩阵,得到基础矩阵F
    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    ///下面进行对基础矩阵F进行约束调整
    //将基础矩阵svd分解
    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    //根据基础矩阵的性质分解出来的w第三个元素应该为0
    w.at<float>(2)=0;

    //返回复合要求的基础矩阵
    return  u*cv::Mat::diag(w)*vt;
}

//通过单应矩阵H21以及H21_inv 反投影,计算误差,得到当前单应矩阵H的分数
float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
{   
    const int N = mvMatches12.size();

    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    //判断通过单应矩阵重投影是否成功的阈值
    const float th = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);

    //遍历所有的匹配点
    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // H约束: 点与点之间的约束

        // Reprojection error in first image
        // x2=[u2,v2,1]^T  x1=[u1,v1,1]^T
        // 这里的 H_21_inv = [h11_inv h12_inv h12_inv
        //                   h21_inv h22_inv h23_inv
        //                   h31_inv h32_inv h33_inv ]
        // H模型: 原模型: x2 = \hat{H_21} * x1
        // 逆模型(左乘\hat{H_21}): H_21^{-1} * x2 = x1
        // 逆模型就是将第2帧的特征点利用H21_inv矩阵反投影到第一帧
        // 理解的时候,可以结合<视觉SLAM14讲-第二版 P171 公式7.20>来看
        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);    //把H矩阵第3行的约束嵌入到第一第二行中,相当于归一化?
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;  //
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        //计算u2，v2投影到F1后与u1,v1的距离的平方，也就是重投影误差
        //H模型几何距离,使用对称转移误差,见<计算机视觉中的多视图几何> P58 公式3.7
        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        //chiSquare1>th说明匹配的点对F1投影到F2，重投影失败
        if(chiSquare1>th)
            bIn = false;
        else
            score += th - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1

        // H模型: 原模型: x2 = \hat{H_21} * x1
        // 步骤基本同上
        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;
	
        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += th - chiSquare2;

	//bIn标志着此对匹配点是否重投影成功
        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

//利用基础矩阵F进行重投影,计算得分
float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = mvMatches12.size();

    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    //std::vector<bool> vbMatchesInliers是输出,输出哪些匹配点对是内点
    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 3.841;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i=0; i<N; i++)  //遍历每一个特征点对
    {
        bool bIn = true;

        //mvMatches12[i]=pair(参考帧特征点idx,当前帧特征点idx) ,这两个特征点认为相互匹配, 现在取出来看是否真的匹配
        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        /****************************************************
         * 利用F重投影
         * (0)由极线约束的示意图可知: 极线l'在图像平面上的法向量可以由
         *    极线上的两点叉乘得到
         * (1)根据公式l'=F*u ==> 极线法向量 l2=F21 * x1  ===>极线法向量(a,b,c)
         * (2)于是可以得到极线点法式方程: a(X-u1)+b(Y-v1)+c=0
         * (3)假设在第一帧的点 x1 =[u1,v1,1]^T
         * (4)根据点到直线距离公式 dist= |a*u1+b*u2+c|/ sqrt(a^2+b^2)
         ****************************************************/

        ///1. 将第一帧的特征点重投影到第二帧,计算误差
        // l2=F21x1=[a2,b2,c2]^T (3x1)列向量
        // F= [ f11 f12 f13
        //      f21 f22 f23
        //      f31 f32 f33 ]
        //
        // f1=[f11 f12 f13] U1=[u1 v1 1]^T  ===> a2=f1*U1
        // 这里计算得到极线向量l2=[a2,b2,c2]
        const float a2 = f11*u1+f12*v1+f13;     // (Fu)_x
        const float b2 = f21*u1+f22*v1+f23;     // (Fu)_y
        const float c2 = f31*u1+f32*v1+f33;
	
        // 计算点x1到直线l2距离
        const float num2 = a2*u2+b2*v2+c2;

        // 距离的平方
        const float squareDist1 = num2*num2/(a2*a2+b2*b2);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        //判断距离是否超过阈值,并计算得分,距离越大,得分越小
        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;


        ///2. 将第二帧的特征点重投影到第一帧,计算误差(思路基本同上,计算点到直线距离)
        // l1 =x2tF21=[a1,b1,c1] (1x3) 行向量
        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        //判断距离是否超过阈值,并计算得分,距离越大,得分越小
        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        //设置标志: 描述这对特征点是否是inlier
        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    //匹配点中
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // Compute Essential Matrix from Fundamental Matrix
    // 从基础矩阵F计算出本质矩阵E (利用内参K)
    cv::Mat E21 = K.t()*F21*K;

    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses (从E分解出4种可能的情况)
    // 从矩阵E分解出R,t ,其中分解出两个R
    DecomposeE(E21,R1,R2,t);  

    // t又可以分为+t和-t
    cv::Mat t1=t;
    cv::Mat t2=-t;

    // Reconstruct with the 4 hyphoteses and check
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
    float parallax1,parallax2, parallax3, parallax4;

    //mvKeys1:第一帧特征点 mvKeys2:第二帧特征点
    //mvMatches12:储存着匹配点对在参考帧F1特征点数组和当前帧F2特征点数组中的序号
    //vbMatchesInliers:有效的匹配点对
    int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

    //取得分最高的
    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();

    //minTriangulated=50
    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);

    //如果有两种情况的R,t得分都差不多,没有明显差别,nsimilar将>1,这次初始化失败
    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    //nsimilar>1表明没有哪个模型明显胜出
    //匹配点三角化重投影成功数过少
    if(maxGood<nMinGood || nsimilar>1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    // 如果某种情况的R,t得分比较高,则取该情况的R,t
    // 并将三角化的点拷贝,R,t拷贝,三角化标志位拷贝
    if(maxGood==nGood1)
    {
	//如果模型一对应的视差角大于最小值
        if(parallax1>minParallax)
        {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}

//从H矩阵恢复出R,t [尚有问题]
bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{

    //N，通过H重投影成功的数量
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988

    
    // 将H矩阵由图像坐标系变换到相机坐标系 ??
    cv::Mat invK = K.inv();
    cv::Mat A = invK*H21*K;

    // 对H矩阵进行SVD, 但是为什么H矩阵要先invK*H21*K ?
    cv::Mat U,w,Vt,V;
    cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
    //vt转置
    V=Vt.t();

    //s是设的一个值
    //cv::determinant(U)为U的行列式
    //因为SVD的U和Vt都是单位正交阵,因此s*s=1, s=+1或-1
    float s = cv::determinant(U)*cv::determinant(Vt);

    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);

    //注意d1>d2>d3
    //看吴博讲解的ppt19页，只考虑d1!=d2!=d3的情况，其他情况返回失败
    if(d1/d2<1.00001 || d2/d3<1.00001)
    {
        return false;
    }

    vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities
    //x1[i],x3[i]: i对应这4种情况的某一种 {e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1}
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};

    //case d'=+d2
    //根据推导的结果
    //可以解得:旋转矩阵R绕y轴旋转的角度 sin(theta)
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);
    //cos(theta)
    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    //遍历4种情况,将4种情况的R,t,n都存起来
    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=ctheta;
        Rp.at<float>(0,2)=-stheta[i];
        Rp.at<float>(2,0)=stheta[i];
        Rp.at<float>(2,2)=ctheta;

        //因为R'=s*U^T*R*V,且s只能等于+1或-1
        //所以R =s*U*R'*Vt
        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        //t'=(d1-d3)*(x1 0 x3)^T ,  因为x3数组中的x3[i]符号与ppt相反了,所以这里带负号是正确的
        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=-x3[i];
        tp*=d1-d3;

        //因为t'=U^T*t
        //所以t =U*t'
        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));


        //n'=[x1,x2,x3]^T
        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        /*********************************
         * 问题
         * 这里 t'=[x1 0 -x3]  (t'符号是对的)
         *     n'=[x1 0  x3]
         *     与ppt对不上
         * [ps:貌似没影响,因为没用到 n' ]
         *********************************/

        //因为n'=V^T*n
        //所以n=V*n'
        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    //case d'=-d2
    //sin(theta)的推导结果与d'=+d2的时候不一样,这里重新创建一个
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);
    //cos(theta)
    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=cphi;
        Rp.at<float>(0,2)=sphi[i];
        Rp.at<float>(1,1)=-1;
        Rp.at<float>(2,0)=sphi[i];
        Rp.at<float>(2,2)=-cphi;

        //因为R'=s*U^T*R*V,且s只能等于+1或-1
        //所以R =s*U*R'*Vt
        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        //因为t'=U^T*t
        //所以t =U*t'
        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=x3[i];
        tp*=d1+d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        //因为n'=V^T*n
        //所以n=V*n'
        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    int bestGood = 0;
    int secondBestGood = 0;    
    int bestSolutionIdx = -1;
    float bestParallax = -1;
    vector<cv::Point3f> bestP3D;
    vector<bool> bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    // 经过上面的计算，总共有8种R，t计算结果，遍历这8种可能模型
    // 通过计算出匹配点的三角化重投影成功的数量，来找出最好模型和次好模型
    for(size_t i=0; i<8; i++)
    {
        float parallaxi;
        vector<cv::Point3f> vP3Di;   //用来储存三角化得到的3D点
        vector<bool> vbTriangulatedi;//三角化是否成功标志
	
        //计算在输入Rt下，匹配点三角化重投影成功的数量
        int nGood = CheckRT(vR[i],vt[i],mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K,vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);

        //储存得分最高的情况的R,t
        if(nGood>bestGood)
        {
            secondBestGood = bestGood;  //原来最高得分的变成次高得分
            bestGood = nGood;
            bestSolutionIdx = i;        //储存最优的R,t对应的索引i
            bestParallax = parallaxi;   //储存视差角
            bestP3D = vP3Di;            //储存三角化得到的3D点
            bestTriangulated = vbTriangulatedi;
        }
        else if(nGood>secondBestGood)
        {
            secondBestGood = nGood;     //如果得分比得分次高的情况高,则本次得分变成次高分,只记录次高分分数即可
        }
    }

    //secondBestGood<0.75*bestGood 如果最好模型与次好模型差距足够大
    //bestParallax>=minParallax 最好模型对应的视差角大于此值
    //bestGood>minTriangulated 最好模型对应的匹配点三角化重投影成功数量大于此阈值
    //bestGood>0.9*N 匹配点三角化重投影成功数量占通过H重投影成功数量的比例需要大于0.9
    //满足以上情况,则认为分解出的R,t是满足的,拷贝R,t,三角化后的点,以及三角化情况,返回
    if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
    {
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        vP3D = bestP3D;
        vbTriangulated = bestTriangulated;

        return true;
    }

    return false;
}

//三角化
void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{   
    /****************************************************
     * 三角化
     * 1.输入:
     *  (1)两帧图像的特征点kp1,kp2
     *  (2)相机位姿P1=K*T[I|0]=[K|0]  P2=K*T[Rcw|tcw] ===> T1  T2
     * 2.输出:
     *      恢复出特征点对应的3D点(世界坐标系)
     * 3.理论推导
     *  (1)<视觉SLAM14讲-第二版P177>
     *  (2)<深蓝学院-手写VIO第6讲>
     ****************************************************/

    //系数矩阵A
    cv::Mat A(4,4,CV_32F);

    //三角化约束方程 (一个位姿可以得到两个方程, 两帧即可求解)
    // (1) [Tn_(第3行)*u1 - Tn_(第一行)]*y=0
    // (2) [Tn_(第3行)*v1 - Tn_(第二行)]*y=0
    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);


    /**************************************
     * 解释
     * (1)在<深蓝学院-手写VIO第6讲>的描述是,对系数矩阵A^TA进行SVD分解,
     * 然后最小二乘解就是最小特征值对应的特征向量
     * (2)这里则直接对系数矩阵A进行SVD分解,为什么?
     *
     * 答:
     *  (1)若 svd(A)==> A=U*(w)*Vt
     *  (2)那么 svd(A^TA)==> A^TA= V[w^T(U^TU)w]Vt = V(w^T*w)Vt
     *  所以, 两种分解都是取Vt的最后一维的行向量,所以结果是一样的
     **************************************/

    //求解齐次线性方程组Ay=0
    //对A进行SVD
    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    //取最后一个行向量(最小奇异值对应的特征向量) 维度是4x1
    x3D = vt.row(3).t();
    //转换为齐次坐标(即得到了特征点在世界坐标系下的3D坐标)
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}


void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    ///归一化变换
    // 见:https://epsilonjohn.gitee.io/2020/02/17/%E7%AC%AC%E4%B8%80%E8%AE%B2-ORB-SLAM2-%E9%A2%84%E5%A4%87%E7%9F%A5%E8%AF%86-1/#d-2d-homographyfundamental-%E5%AF%B9%E6%9E%81%E5%87%A0%E4%BD%95
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    // 这是要输出的归一化之后的点
    vNormalizedPoints.resize(N);

    //求平均值
    for(int i=0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    meanX = meanX/N;
    meanY = meanY/N;

    float meanDevX = 0;
    float meanDevY = 0;

    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    // 尺度缩放
    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    // 构建归一化矩阵
    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}

//输入R,t,以及匹配的特征点,计算对应3D点在相机前的距离,和视差角,计算R,t得分
int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    //vKeys1:第一帧特征点
    //vKeys2:第二帧特征点
    //vMatches12:储存着匹配点对在参考帧F1特征点数组和当前帧F2特征点数组中的序号
    //vbMatchesInliers:有效的匹配点对(内点)

    // 内参
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    //初始化
    vbGood = vector<bool>(vKeys1.size(),false);
    vP3D.resize(vKeys1.size());

    vector<float> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K*[I|0]
    // 相机1的投影矩阵P1=K*[I|0]_(3x4)=[K|0]，世界坐标系和相机1坐标系相同
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3));   //将内参矩阵K复制到P1的左上角3x3

    // 相机1的光心在世界坐标系坐标(相机1坐标系原点)
    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    // Camera 2 Projection Matrix K*[R|t]
    // R=R2<--1
    // 设相机2位姿T2=[R|t]
    // 相机2投影矩阵P2=K*T2_(3x4)=K*[R|t]_(3x4) 最终得到一个3x4的投影矩阵
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;

    // 相机2的光心在世界坐标系坐标
    // 因为[R,t]描述的是将世界坐标系的点转换到相机坐标系,即pc=Rcw*pw+tcw
    // 所以,将相机坐标系2的点pc(0,0)转换到世界坐标系,则有pw=Rwc(pc-tcw)=Rcw^T*(pc-tcw)
    cv::Mat O2 = -R.t()*t;

    int nGood=0;

    //遍历所有的匹配点
    for(size_t i=0, iend=vMatches12.size();i<iend;i++)
    {
        //如果是内点,才往下进行,否则就跳过
        if(!vbMatchesInliers[i])
            continue;

        //取匹配点
        const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
        const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
	
        //3d点在相机1坐标系(世界坐标系)下的坐标
        cv::Mat p3dC1;
	
        //三角化,得到当前匹配点对对应的在世界坐标系下的3D点坐标p3dC1
        //由于世界坐标系和相机1坐标系重合，所以p3dC1同时也是匹配点对应的空间点在相机1坐标系中的坐标
        Triangulate(kp1,kp2,P1,P2,p3dC1);

        //isfinite()判断一个浮点数是否是一个有限值
        //相当于是确定p3dC1前三位数值正常
        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[vMatches12[i].first]=false;
            continue;
        }

        // Check parallax
        //normal1是相机1到3d点的向量
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = cv::norm(normal1);
	
        //normal2是相机2到3d点的向量
        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);

        //cosParallax为视差角的余弦，也就是normal1与normal2的余弦 余弦公式
        float cosParallax = normal1.dot(normal2)/(dist1*dist2);

        // Check depth in front of first camera 检查3D点在相机1前面的深度
        // (only if enough parallax, as "infinite" points can easily go to negative depth)
        //p3dC1.at<float>(2)<=0说明3d点在光心后面，深度为负
        //p3dC1视差角较大，且深度为负则淘汰,则不计入得分
        if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        // 检查3D点在相机2前面的深度
        cv::Mat p3dC2 = R*p3dC1+t;  //先把3D点转换到相机2坐标系下
	
        //p3dC2视差角较大，且深度为负则则不计入得分
        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check reprojection error in first image
        // 计算3D点在第一个图像上的重投影误差
        float im1x, im1y;
        float invZ1 = 1.0/p3dC1.at<float>(2);
        // 相机模型
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;
	
        //平方误差
        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);

        //阈值判断,误差太大,则不计入得分
        if(squareError1>th2)
            continue;

        // Check reprojection error in second image
        // 计算3D点在第二帧图像的重投影误差
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2.at<float>(2);
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

	
        // 重投影误差太大，则不计入得分
        if(squareError2>th2)
            continue;

        //到这里说明这对匹配点三角化重投影成功了
        //vMatches12[i].first 第一帧(参考帧)特征点的索引
        vCosParallax.push_back(cosParallax);
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
        //这一对R,t对应的得分+1
        nGood++;

        //当视差角足够大,vbGood[idx]设置为true
        if(cosParallax<0.99998)
            vbGood[vMatches12[i].first]=true;
    }

    if(nGood>0)
    {
        //将视差角余弦有小到大排序
        sort(vCosParallax.begin(),vCosParallax.end());

        //取出第50个，或者最后那个也就是最大那个
        size_t idx = min(50,int(vCosParallax.size()-1));
        //计算出视差角(转换成角度)
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;

    //返回这对R,t的得分,以及对应的视差角
    return nGood;
}

//分解本质矩阵E
void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    ///分解本质矩阵E: 具体公式见<视觉SLAM14讲-第二版 P169 公式7.15>
    //SVD分解
    cv::Mat u,w,vt;
    cv::SVD::compute(E,w,u,vt);


    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    //W是沿z轴旋转90度得到的旋转矩阵
    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    //对应公式
    R1 = u*W*vt;
    if(cv::determinant(R1)<0)
        R1=-R1;

    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;

    //得到两个R
}

} //namespace ORB_SLAM
