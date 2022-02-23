#include<opencv2\opencv.hpp>  
#include<opencv2\face.hpp>
#include<opencv2\core\core.hpp>
#include<opencv2\face\facerec.hpp>
#include <fstream>  
#include <sstream> 
#include<math.h>
#include<vector>
#include <opencv2/imgproc/types_c.h>
#include<opencv2\highgui.hpp>
#include<opencv2\imgproc.hpp>
#include <iostream> 
#include <stdio.h>
#include<string>
#include<direct.h>
#include<opencv2\dnn\dnn.hpp>
#include <memory>
#include <cstdlib>
#include <dlib/opencv.h>  
#include <dlib/image_processing/frontal_face_detector.h>  
#include <dlib/image_processing/render_face_detections.h>  
#include <dlib/image_processing.h>  
#include <dlib/gui_widgets.h>  
#include <windows.h>
using namespace cv::dnn;
//using namespace dlib;
using namespace std;
using namespace cv;
using namespace cv::face;

RNG g_rng(12345);
Ptr<FaceRecognizer> model;
#define DETECT_BUFFER_SIZE 0x20000
const size_t inWidth = 300; //output image size
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const Scalar meanVal(104.0, 117.0, 123.0);
struct faceRoi {
	int x;
	int y;
	int w;
	int h;
	int confidence;
	cv::Mat cropImg;
	cv::Point2f landmarks[5];
};


static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file)
	{
		string error_message = "No valid input file was given, please check the given filename.";
		//CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line))
	{
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty())
		{
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int Predict(Mat src_image)  //识别人脸
{
	Mat face_test;
	int predict = 0;
	//截取的ROI人脸尺寸调整
	if (src_image.rows >= 120)
	{
		//改变图像大小，使用双线性差值
		resize(src_image, face_test, Size(92, 112));

	}
	//判断是否正确检测ROI
	if (!face_test.empty())
	{
		//测试图像应该是灰度图  
		predict = model->predict(face_test);
	}
	cout << predict << endl;
	return predict;
}
int CreateMode()
{
	string fn_csv = "D:\\opencv_stuff\\ORL_92x112\\csv.txt";
	// 2个容器来存放图像数据和对应的标签
	vector<Mat> images;
	vector<int> labels;
	// 读取数据. 如果文件不合法就会出错
	// 输入的文件名已经有了.
	try
	{
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e)
	{
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// 文件有问题，我们啥也做不了了，退出了
		exit(1);
	}
	// 如果没有读取到足够图片，也退出.
	if (images.size() <= 1)
	{
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		//	CV_Error(CV_StsError, error_message);
	}
	for (size_t i = 0; i < images.size(); i++)
	{
		if (images.at(i).type() != CV_8UC1)
		{
			std::cerr << "图像的类型必须为CV_8UC1!" << endl;
			return 0;
		}
	}

	//检测尺寸等于正样本尺寸第一张的尺寸
	Size positive_image_size = images[0].size();
	cout << "正样本的尺寸是:" << positive_image_size << endl;
	//遍历所有样品，检测尺寸是否相同
	for (size_t i = 0; i < images.size(); i++)
	{
		if (positive_image_size != images[i].size())
		{
			std::cerr << "所有的样本的尺寸大小不一，请重新调整好样本大小！按任意键退出" << endl;
			cout << i;
			waitKey(0);
			return 0;
		}

	}
	// 下面的几行代码仅仅是从你的数据集中移除最后一张图片
	//[gm:自然这里需要根据自己的需要修改，他这里简化了很多问题]
	Mat testSample = images[images.size() - 2];
	int testLabel = labels[labels.size() - 2];
	images.pop_back();
	labels.pop_back();
	// 下面几行创建了一个特征脸模型用于人脸识别，
	// 通过CSV文件读取的图像和标签训练它。
	// T这里是一个完整的PCA变换
	//如果你只想保留10个主成分，使用如下代码
	//      cv::createEigenFaceRecognizer(10);
	//
	// 如果你还希望使用置信度阈值来初始化，使用以下语句：
	//      cv::createEigenFaceRecognizer(10, 123.0);
	//
	cv::face::LBPHFaceRecognizer::create(10, 123.0);

	Ptr<FaceRecognizer> model = LBPHFaceRecognizer::create();
	model->train(images, labels);
	model->save("D:\\opencv_stuff\\trained\\MyFaceLBPHModel.xml");

	// 下面对测试图像进行预测，predictedLabel是预测标签结果
	int predictedLabel = model->predict(testSample);

	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	cout << result_message << endl;
	cout << "training successed" << endl;
}
string OpenCamera(int choice)//choice表示需要该函数进行哪种操作，若参数为1则只检测一次人脸判断是否通过（权限检测）,2为循环检测
{
	VideoCapture cap(0);    //打开默认摄像头  
	if (!cap.isOpened())
	{
		return "error";
	}
	Mat frame;
	Mat gray;
	//这个分类器是人脸检测所用
	CascadeClassifier cascade;
	bool stop = false;
	//训练好的文件名称，放置在可执行文件同目录下  
	cascade.load("D:\\opencv_stuff\\haarcascade_frontalface_alt2.xml");//感觉用lbpcascade_frontalface效果没有它好，注意哈！要是正脸

	model = LBPHFaceRecognizer::create();
	//1.加载训练好的分类器
	model->read("D:\\opencv_stuff\\trained\\MyFaceLBPHModel.xml");// opencv2用load
																  //3.利用摄像头采集人脸并识别
	int times = 0,defalt=0;
	string name;
	while (1)
	{
		cap >> frame;

		vector<Rect> faces(0);//建立用于存放人脸的向量容器

		cvtColor(frame, gray, CV_RGB2GRAY);//测试图像必须为灰度图

		equalizeHist(gray, gray); //变换后的图像进行直方图均值化处理  
								  //检测人脸
								  //	cascade.detectMultiScale(gray, faces, 1.1, 4, 0
								  //|CV_HAAR_FIND_BIGGEST_OBJECT  
								  //		| CV_HAAR_DO_ROUGH_SEARCH,
								  //| CV_HAAR_SCALE_IMAGE,
								  //		Size(30, 30), Size(500, 500));
		cascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30), Size(500, 500));
		Mat* pImage_roi = new Mat[faces.size()];    //定以数组
		Mat face;
		Point text_lb;//文本写在的位置
					  //框出人脸
		string str;
		for (int i = 0; i < faces.size(); i++)
		{
			pImage_roi[i] = gray(faces[i]); //将所有的脸部保存起来
			text_lb = Point(faces[i].x, faces[i].y);
			if (pImage_roi[i].empty())
				continue;
			switch (Predict(pImage_roi[i])) //对每张脸都识别
			{
			case 35:str = "cyh"; break;
			case 41:str = "sun shibo"; break;
			case 42:str = "zhang xiangying"; break;
			case 43:str = "yang zhiheng"; break;
			default:str = "error"; break;
			}
			if (choice == 4||choice==5)
			{
				if (times == 0)
				{
					name = str;
				}
				if (name == str) ++times;
				else
				{
					times = 0;
					++defalt;
				}
				if (defalt > 5)
				{
					destroyWindow("face");
					return "error";
				}
				if (times == 5)
				{
					destroyWindow("face");
					return name;
				}
			}
				Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));//所取的颜色任意值
				rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), color, 1, 8);//放入缓存
				putText(frame, str, text_lb, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));//添加文字
			}

			delete[]pImage_roi;
			imshow("face", frame);
			waitKey(200);
		}
	}

	int NewFace()//添加新的人脸信息，待改
	{
		CascadeClassifier cascada;
		cascada.load("D:\\opencv_stuff\\haarcascade_frontalface_alt2.xml");
		VideoCapture cap(0);
		Mat frame, myFace;
		int pic_num = 1;
		int num = 0;//num保存读到的序号
		FILE *file;//以只读方式打开csv
		FILE *fil;//以读写方式打开csv
		int back = 0;
		file = fopen("D:\\opencv_stuff\\ORL_92x112\\csv.txt", "r");
		fseek(file, back, SEEK_END);
		char c = 0;//读file
		while (c != ';')
		{
			--back;
			fseek(file, back, SEEK_END);
			c = getc(file);
		}
		c = getc(file);
		num = c - '0';
		c = getc(file);
		while (c != EOF&&c <= '9'&&c >= '0')
		{
			num *= 10;
			num = num + c - '0';
			c = getc(file);
		}
		num = num + 2;
		std::string Filenam = format("D:\\opencv_stuff\\ORL_92x112\\s%d", num, num);
		const char* Filename = Filenam.c_str();
		_mkdir(Filename);//创建新文件夹
		fclose(file);
		fil = fopen("D:\\opencv_stuff\\ORL_92x112\\csv.txt", "r+");
		fseek(file, 0, SEEK_END);
		while (1) {
			//摄像头读图像
			cap >> frame;
			vector<Rect> faces;//vector容器存检测到的faces
			Mat frame_gray;
			cvtColor(frame, frame_gray, COLOR_BGR2GRAY);//转灰度化，减少运算
			cascada.detectMultiScale(frame_gray, faces, 1.1, 4, 4, Size(70, 70), Size(1000, 1000));
			printf("检测到人脸个数：%d\n", faces.size());
			//1.frame_gray表示的是要检测的输入图像 2.faces表示检测到的人脸目标序列,3. 1.1表示每次图像尺寸减小的比例
			//4. 4表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸
			/*5.flags–要么使用默认值，要么使用CV_HAAR_DO_CANNY_PRUNING，
			函数将会使用Canny边缘检测来排除边缘过多或过少的区域，
			因为这些区域通常不会是人脸所在区域；opencv3 以后都不用这个参数了*/
			//6. Size(100, 100)为目标的最小尺寸 一般为30*30 是最小的了 也够了
			//7. Size(500, 500)为目标的最大尺寸 其实可以不用这个，opencv会自动去找这个最大尺寸
			//适当调整5,6,7两个参数可以用来排除检测结果中的干扰项。
			//识别到的脸用矩形圈出
			for (int i = 0; i < faces.size(); i++)
			{
				rectangle(frame, faces[i], Scalar(255, 0, 0), 2, 8, 0);
			}
			//当只有一个人脸时，开始拍照
			if (faces.size() == 1)
			{
				Mat faceROI = frame_gray(faces[0]);//在灰度图中将圈出的脸所在区域裁剪出
												   //cout << faces[0].x << endl;//测试下face[0].x
				resize(faceROI, myFace, Size(92, 112));//将兴趣域size为92*112
				putText(frame, to_string(pic_num), faces[0].tl(), 3, 1.2, (0, 0, 225), 2, 0);//在 faces[0].tl()的左上角上面写序号
				string filename = format("D:\\opencv_stuff\\ORL_92x112\\s%d\\s%d_%d.jpg", num, num, pic_num); //存放在当前项目文件夹以1-10.jpg 命名，format就是转为字符串
				string csvnam = format("\nD:\\opencv_stuff\\ORL_92x112\\s%d\\s%d_%d.jpg;%d", num, num, pic_num, num - 1); //存放在当前项目文件夹以1-10.jpg 命名，format就是转为字符串
				const char* csvname = csvnam.c_str();
				imwrite(filename, myFace);//存在当前目录下
				imshow(filename, myFace);//显示下size后的脸
				fputs(csvname, fil);
				waitKey(500);//等待500us
				destroyWindow(filename);//:销毁指定的窗口
				pic_num++;//序号加1
				if (pic_num == 11)
				{
					return 0;//当序号为11时退出循环
				}
			}
			int c = waitKey(10);
			if ((char)c == 27) { break; } //10us内输入esc则退出循环
			imshow("frame", frame);//显示视频流
			waitKey(200);//等待100us
		}
		fclose(fil);
		return 0;
	}
int SaveExpression(string name)
{
	int facepoint_start = 0;
	int save = 0;
	float min_confidence = 0.8;
	FILE * fil;
	String modelConfiguration = "./face_detector/deploy.prototxt";
	String modelBinary = "./face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel";
	int maxw = 0, minw = 0, maxl = 0, minl = 0;
	int w, l;//w=maxw-minw,l=maxl-minl;
	std::vector<String> files;
	std::vector<faceRoi> faceList;
	dlib::shape_predictor pose_model;
	dlib::deserialize("./face_detector/shape_predictor_68_face_landmarks.dat") >> pose_model;

	//! [Initialize network]
	dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);//Reads a network model stored in Caffe model in memory
	if (net.empty()) {
		exit(-1);
	}

	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cout << "Couldn't open camera : " << endl;
		return -1;
	}
	cout << "按q开始保存或识别表情";
	fflush(stdin);
	while (1)   //for (;;)死循环
	{
		Mat frame;
		cap >> frame;

		if (frame.empty()) {
			break;
		}
		if (frame.channels() == 4)
			cvtColor(frame, frame, COLOR_BGRA2BGR);

		//! [Prepare blob]
		Mat inputBlob = blobFromImage(frame, inScaleFactor,
			Size(inWidth, inHeight), meanVal, false, false); //Convert Mat to batch of images
		net.setInput(inputBlob, "data"); //set the network input
		Mat detection = net.forward("detection_out");   //! [Make forward pass]

		std::vector<double> layersTimings;
		double freq = getTickFrequency() / 1000;//用于返回CPU的频率。get Tick Frequency。这里的单位是秒，也就是一秒内重复的次数。
		double time = net.getPerfProfile(layersTimings) / freq;

		Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());//101*7矩阵

		ostringstream ss;
		ss << "FPS: " << 1000 / time << " ; time: " << time << " ms";
		putText(frame, ss.str(), Point(20, 20), 0, 0.5, Scalar(0, 0, 255));

		float confidenceThreshold = min_confidence;
		for (int i = 0; i < detectionMat.rows; i++)
		{
			float confidence = detectionMat.at<float>(i, 2);      //第二列存放可信度

			if (confidence > confidenceThreshold)//满足阈值条件
			{
				//存放人脸所在的图像中的位置信息
				int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
				int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
				int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
				int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

				Rect object((int)xLeftBottom, (int)yLeftBottom,//定义一个矩形区域（x,y,w,h)
					(int)(xRightTop - xLeftBottom),
					(int)(yRightTop - yLeftBottom));
				cv::rectangle(frame, object, Scalar(0, 255, 0));//画个边框
																//使用dlib进行人脸特征点检测
				dlib::rectangle face((int)xLeftBottom, (int)yLeftBottom, (int)(xRightTop), (int)(yRightTop));
				dlib::cv_image<dlib::bgr_pixel> cimg(frame);
				std::vector<dlib::full_object_detection> shapes;
				shapes.push_back(pose_model(cimg, face));
				if (!shapes.empty()) {
					if (waitKey(100) == 'q')
						save = 1;
					for (int i = facepoint_start; i < 68; i++) {
						circle(frame, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), 2, cv::Scalar(0, 255, 0), -1);
						if (save == 1)
						{
							if (i <= 26)
							{
								if (i == 0)
								{
									string plac = "D:\\opencv_stuff\\expression\\face\\"+name+".txt";
									const char* place = plac.c_str();
									fil = fopen(place, "w+");
									fseek(fil, 12, SEEK_SET);
									maxw = shapes[0].part(i).x();
									minw = shapes[0].part(i).x();
									minl = shapes[0].part(i).y();
									maxl = shapes[0].part(i).y();
								}
								if (shapes[0].part(i).x() > maxw) maxw = shapes[0].part(i).x();
								else if (shapes[0].part(i).x() < minw) minw = shapes[0].part(i).x();
								if (shapes[0].part(i).y() > maxl) maxl = shapes[0].part(i).y();
								else if (shapes[0].part(i).y() < minl) minl = shapes[0].part(i).y();
								fprintf(fil, "%03d%03d", shapes[0].part(i).x(), shapes[0].part(i).y());
								if (i == 26)
								{
									l = maxl - minl;
									w = maxw - minw;
									fseek(fil, 0, SEEK_SET);
									fprintf(fil, "%03d%03d%03d%03d", l,w,minl,minw);
									fclose(fil);
								}
							}
							else if (i > 26 && i <= 30)
							{
								if (i == 27)
								{
									string plac = "D:\\opencv_stuff\\expression\\noseup\\" + name + ".txt";
									const char* place = plac.c_str();
									fil = fopen(place, "w+");
									fseek(fil, 12, SEEK_SET);
									maxw = shapes[0].part(i).x();
									minw = shapes[0].part(i).x();
									minl = shapes[0].part(i).y();
									maxl = shapes[0].part(i).y();
								}
								if (shapes[0].part(i).x() > maxw) maxw = shapes[0].part(i).x();
								else if (shapes[0].part(i).x() < minw) minw = shapes[0].part(i).x();
								if (shapes[0].part(i).y() > maxl) maxl = shapes[0].part(i).y();
								else if (shapes[0].part(i).y() < minl) minl = shapes[0].part(i).y();
								fprintf(fil, "%03d%03d", shapes[0].part(i).x(), shapes[0].part(i).y());
								if (i == 30)
								{
									l = maxl - minl;
									w = maxw - minw;
									fseek(fil, 0, SEEK_SET);
									fprintf(fil, "%03d%03d%03d%03d", l, w, minl, minw);
									fclose(fil);
								}
							}
							else if (i > 30 && i <= 35)
							{
								if (i == 31)
								{
									string plac = "D:\\opencv_stuff\\expression\\nosedown\\" + name + ".txt";
									const char* place = plac.c_str();
									fil = fopen(place, "w+");
									fseek(fil, 12, SEEK_SET);
									maxw = shapes[0].part(i).x();
									minw = shapes[0].part(i).x();
									minl = shapes[0].part(i).y();
									maxl = shapes[0].part(i).y();
								}
								if (shapes[0].part(i).x() > maxw) maxw = shapes[0].part(i).x();
								else if (shapes[0].part(i).x() < minw) minw = shapes[0].part(i).x();
								if (shapes[0].part(i).y() > maxl) maxl = shapes[0].part(i).y();
								else if (shapes[0].part(i).y() < minl) minl = shapes[0].part(i).y();
								fprintf(fil, "%03d%03d", shapes[0].part(i).x(), shapes[0].part(i).y());
								if (i == 35)
								{
									l = maxl - minl;
									w = maxw - minw;
									fseek(fil, 0, SEEK_SET);
									fprintf(fil, "%03d%03d%03d%03d", l, w, minl, minw);
									fclose(fil);
								}
							}
							if (i > 35 && i <= 41)
							{
								if (i == 36)
								{
									string plac = "D:\\opencv_stuff\\expression\\leye\\" + name + ".txt";
									const char* place = plac.c_str();
									fil = fopen(place, "w+");
									fseek(fil, 12, SEEK_SET);
									maxw = shapes[0].part(i).x();
									minw = shapes[0].part(i).x();
									minl = shapes[0].part(i).y();
									maxl = shapes[0].part(i).y();
								}
								if (shapes[0].part(i).x() > maxw) maxw = shapes[0].part(i).x();
								else if (shapes[0].part(i).x() < minw) minw = shapes[0].part(i).x();
								if (shapes[0].part(i).y() > maxl) maxl = shapes[0].part(i).y();
								else if (shapes[0].part(i).y() < minl) minl = shapes[0].part(i).y();
								fprintf(fil, "%03d%03d", shapes[0].part(i).x(), shapes[0].part(i).y());
								if (i == 41)
								{
									l = maxl - minl;
									w = maxw - minw;
									fseek(fil, 0, SEEK_SET);
									fprintf(fil, "%03d%03d%03d%03d", l, w, minl, minw);
									fclose(fil);
								}
							}
							if (i > 41 && i <= 47)
							{
								if (i == 42)
								{
									string plac = "D:\\opencv_stuff\\expression\\reye\\" + name + ".txt";
									const char* place = plac.c_str();
									fil = fopen(place, "w+");
									fseek(fil, 12, SEEK_SET);
									maxw = shapes[0].part(i).x();
									minw = shapes[0].part(i).x();
									minl = shapes[0].part(i).y();
									maxl = shapes[0].part(i).y();
								}
								if (shapes[0].part(i).x() > maxw) maxw = shapes[0].part(i).x();
								else if (shapes[0].part(i).x() < minw) minw = shapes[0].part(i).x();
								if (shapes[0].part(i).y() > maxl) maxl = shapes[0].part(i).y();
								else if (shapes[0].part(i).y() < minl) minl = shapes[0].part(i).y();
								fprintf(fil, "%03d%03d", shapes[0].part(i).x(), shapes[0].part(i).y());
								if (i == 47)
								{
									l = maxl - minl;
									w = maxw - minw;
									fseek(fil, 0, SEEK_SET);
									fprintf(fil, "%03d%03d%03d%03d", l, w, minl, minw);
									fclose(fil);
								}
							}
							if (i > 47)
							{
								if (i == 48)
								{
									string plac = "D:\\opencv_stuff\\expression\\mouth\\" + name + ".txt";
									const char* place = plac.c_str();
									fil = fopen(place, "w+");
									fseek(fil, 12, SEEK_SET);
									maxw = shapes[0].part(i).x();
									minw = shapes[0].part(i).x();
									minl = shapes[0].part(i).y();
									maxl = shapes[0].part(i).y();
								}
								if (shapes[0].part(i).x() > maxw) maxw = shapes[0].part(i).x();
								else if (shapes[0].part(i).x() < minw) minw = shapes[0].part(i).x();
								if (shapes[0].part(i).y() > maxl) maxl = shapes[0].part(i).y();
								else if (shapes[0].part(i).y() < minl) minl = shapes[0].part(i).y();
								fprintf(fil, "%03d%03d", shapes[0].part(i).x(), shapes[0].part(i).y());
								if (i == 67)
								{
									l = maxl - minl;
									w = maxw - minw;
									fseek(fil, 0, SEEK_SET);
									fprintf(fil, "%03d%03d%03d%03d", l, w, minl, minw);
									fclose(fil);
								}
							}
						}
					}
				}

				ss.str("");
				ss << confidence;
				String conf(ss.str());
				String label = "Face: " + conf;
				int baseLine = 0;
				Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
				cv::rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
					Size(labelSize.width, labelSize.height + baseLine)),
					Scalar(255, 255, 255), cv::FILLED);
				putText(frame, label, Point(xLeftBottom, yLeftBottom),
					FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
			}
		}
		cv::imshow("detections", frame);
		if (save == 1) break;
	}
	//destroyWindow("detections");
	return 0;
}
string Expression(int choice)
{
	string name;
	name=OpenCamera(choice);
	if (name == "error")
	{
		cout << "验证失败或者你没有权限！按任意键退出";
		waitKey(0);
		return 0;
	}
	else
	{
		if (choice == 4)
		{
			SaveExpression(name);
		}
		else if (choice == 5)
		{
			SaveExpression(name+"test");
		}
		return name;
	}
}
int Try(FILE *pass, FILE *test, int num,int kind)
{
	int  minl1, minl2, minw1, minw2, l1, l2, w1, w2,samedot=0,i=0;//fprintf(fil, "%d%d%d%d", l,w,minl,minw);
	double mutil, mutiw;
	double dif;
	char c;
	int dot1x, dot1y, dot2x, dot2y;
	fscanf(pass, "%3d", &l1);
	fscanf(pass, "%3d", &w1);
	fscanf(pass, "%3d", &minl1);
	fscanf(pass, "%3d", &minw1);
	fscanf(test, "%3d", &l2);
	fscanf(test, "%3d", &w2);
	fscanf(test, "%3d", &minl2);
	fscanf(test, "%3d", &minw2);
	mutiw = double(w1) / w2;
	mutil = double(l1) / l2;
	for (int i = 0; i < num; ++i)
	{
		fscanf(pass, "%3d", &dot1x);
		fscanf(pass, "%3d", &dot1y);
		fscanf(test, "%3d", &dot2x);
		fscanf(test, "%3d", &dot2y);
		dot1x -= minw1;
		dot1y -= minl1;
		dot2x -= minw2;
		dot2y -= minl2;
		if (kind == 1)
		{
			dif = fabs(dot1y - dot2y*mutil) + 0.1;
			if (dif <= l1 / 30.0)
				++samedot;
		}
		else if (kind == 2)
		{
			dif = fabs(dot1x - dot2x*mutiw) + 0.1;
			if (dif <= w1 / 30.0)
				++samedot;
		}
		else if(kind==3||kind==4)
		{
			dif = (fabs(dot1x - dot2x*mutiw) + 0.1)*(fabs(dot1y - dot2y*mutil) + 0.1);
			if (dif <= (l1*w1 / 1000.0))
				++samedot;
		}
		else if (kind == 5)
		{
			dif = (fabs(dot1x - dot2x*mutiw) + 0.1)*(fabs(dot1y - dot2y*mutil) + 0.1);
			if (dif <= (l1*w1 / 1500.0))
				++samedot;
		}
		else 
		{
			dif = (fabs(dot1x - dot2x*mutiw) + 1)*(fabs(dot1y - dot2y*mutil) + 1);
			if (dif <= (l1*w1 / 1500.0))
				++samedot;
		}
	}
	if (samedot > num*0.80||(num-samedot)<=2 ) return 1;
	else return 0;
}
int Scure(string name)
{
	FILE *pass;
	FILE *test;
	string plac;
	string plac2 = "D:\\opencv_stuff\\expression\\face\\" + name + "test.txt";
	int next = 1;
	int right = 0;
	int num[] = { 27,4,5,6,6,20 };
	string part[] = { "face","noseup","nosedown","leye","reye","mouth" };
	for (int i = 0; i < 6&&right<2; ++i)
	{
		plac = "D:\\opencv_stuff\\expression\\"+part[i]+"\\" + name + ".txt";
		const char* place = plac.c_str();
		pass = fopen(place, "r");
		plac2 = "D:\\opencv_stuff\\expression\\" + part[i] + "\\" + name + "test.txt";
		const char* place2 = plac2.c_str();
		test = fopen(place2, "r");
		next=Try(pass, test, num[i],i);
		if (next == 0) ++right;
	}

	if (right<2) return 1;
	else return 0;
}
int main()
{
	cout << "1.进行人脸检测" << endl << "2.导入新的人脸数据" << endl << "3.训练xml" << endl << "4.导入表情密码" << endl << "5.表情识别" << endl;
	int choice;
	string name;
	cin >> choice;
	system("CLS");
	if (choice == 1)
	{
		OpenCamera(1);
	}
	else if (choice == 2)
	{
		NewFace();

	}
	else if (choice == 3)
	{
		CreateMode();
	}
	else if (choice == 4)
	{
		Expression(4);
		cout << "保存成功,按任意键退出" << endl;
		waitKey(0);
		return 0;
	}
	else if (choice == 5)
	{
		name=Expression(5);
		system("CLS");
		if (Scure(name) == 1)
		{
			cout << "验证成功,按k键退出" << endl;
		}
		else cout << "验证失败,按k键退出";
		fflush(stdin);
		while (1)
		{
			if (waitKey(10) == 'k')
				return 0;
		}
	}
	else
	{
		cout << "输入选项错误！按任意键退出";
		waitKey(0);
	}
	return 0;
}
