#include <opencv2/face.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdio.h>

using namespace std;

namespace fs = filesystem;

/** Function Headers */
void detectAndDisplay(cv::Mat                                        frame,
                      const cv::Ptr<cv::face::FisherFaceRecognizer>& model,
                      int imgWidth, int imgHeight);

/** Global variables */
constexpr static char face_cascade_name[] =
    "../resources/haarcascade_frontalface_alt.xml";
constexpr static char csv_path[] = "../resources/csv.ext";

static cv::CascadeClassifier  face_cascade;
static cv::CascadeClassifier  eyes_cascade;
static string                 window_name = "Capture - Face detection";
static cv::RNG                rng(12345);
static const map<int, string> predictionLabels = {{4, "Morgan Freeman"},
                                                  {5, "std"}};

static void read_csv(const string& fileName, vector<cv::Mat>& images,
                     vector<int>& labels, char separator = ';')
{
  ifstream file(fileName);
  if (!file.is_open())
  {
    throw std::runtime_error("file doesn't exist");
  }

  string line;
  string path;
  string classLabel;

  while (getline(file, line))
  {
    stringstream lines(line);
    getline(lines, path, separator);
    getline(lines, classLabel);

    if (!fs::exists(path) || classLabel.empty())
    {
      throw std::runtime_error("path or class label wrong");
    }

    images.emplace_back(cv::imread(path, cv::ImreadModes::IMREAD_COLOR));
    labels.emplace_back(atoi(classLabel.c_str()));
  }
}

/** @function main */
int main(int argc, const char** argv)
{
  //-- 1. Load the cascades
  if (!face_cascade.load(face_cascade_name))
  {
    printf("--(!)Error loading\n");
    return -1;
  };

  //-- 2. Train the Fisher model
  vector<cv::Mat> images;
  vector<int>     labels;
  read_csv(csv_path, images, labels);

  const auto imageWidth  = images[0].cols;
  const auto imageHeight = images[0].rows;

  auto model = cv::face::FisherFaceRecognizer::create();
  model->train(images, labels);

  while (true)
  {
    auto frame = cv::imread("../resources/training_faces/s4/2.jpg");
    //-- 3. Apply the classifier to the frame
    if (frame.empty())
    {
      printf(" --(!) No captured frame -- Break!");
      break;
    }

    detectAndDisplay(frame, model, imageWidth, imageHeight);
    int c = cv::waitKey(10);
    if ((char)c == 'c')
    {
      break;
    }
  }

  return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(cv::Mat                                        frame,
                      const cv::Ptr<cv::face::FisherFaceRecognizer>& model,
                      int imgWidth, int imgHeight)
{
  std::vector<cv::Rect> faces;
  cv::Mat               frame_gray;
  cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
  equalizeHist(frame_gray, frame_gray);
  //-- Detect faces
  face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | 1,
                                cv::Size(30, 30));
  for (const auto face : faces)
  {
    auto    faceGray = frame_gray(face);
    cv::Mat faceResized;
    cv::resize(faceGray, faceResized, cv::Size(imgWidth, imgHeight),
               cv::InterpolationFlags::INTER_CUBIC);

    int    predictLabel;
    double confidence;
    model->predict(faceResized, predictLabel, confidence);
    cv::rectangle(frame, face, CV_RGB(0, 255, 0), 1);

    auto message =
        predictionLabels.at(predictLabel) + "/" + std::to_string(confidence);
    cout << message << endl;

    cv::putText(frame, message, cv::Point(0, 20),
                cv::HersheyFonts::FONT_HERSHEY_PLAIN, 1.0,
                cv::Scalar(0, 255, 0), 2.0);
  }
  //-- Show what you got
  imshow(window_name, frame);
}