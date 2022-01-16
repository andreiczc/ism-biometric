#include <opencv2/face.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdio.h>
#include <vector>

using namespace std;

namespace fs = filesystem;

/** Function Headers */
void detectAndDisplay(cv::Mat                                      frame,
                      const cv::Ptr<cv::face::LBPHFaceRecognizer>& model);

/** Global variables */
constexpr static char face_cascade_name[] =
    "../resources/haarcascade_frontalface_alt.xml";
constexpr static char csv_path[] = "../resources/csv.ext";

static cv::CascadeClassifier  face_cascade;
static string                 window_name = "Capture - Face detection";
static cv::RNG                rng(12345);
static const map<int, string> predictionLabels = {{4, "Morgan Freeman"},
                                                  {5, "Daniel Craig"},
                                                  {6, "Keanu Reeves"},
                                                  {7, "Kanye West"}};
static const vector<string>   validatePictures = {
    "../resources/training_faces/s4/6.jpg",
    "../resources/validate_faces/s5/1.png",
    "../resources/training_faces/s5/20.jpg",
    "../resources/training_faces/s6/2.jpg",
    "../resources/training_faces/s7/3.jpg",
};

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

static vector<cv::Mat> toGrayscale(const vector<cv::Mat>& images)
{
  vector<cv::Mat> result;
  for (const auto image : images)
  {
    cv::Mat frameGray;
    cv::resize(image, frameGray, cv::Size(250, 250),
               cv::InterpolationFlags::INTER_CUBIC);

    cvtColor(frameGray, frameGray, cv::COLOR_BGR2GRAY);
    equalizeHist(frameGray, frameGray);

    result.emplace_back(frameGray);
  }

  return result;
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

  //-- 2. Train the model
  vector<cv::Mat> images;
  vector<int>     labels;
  read_csv(csv_path, images, labels);
  const auto imagesGrayscale = toGrayscale(images);
  auto       modelLbph       = cv::face::LBPHFaceRecognizer::create(1, 8, 8, 5);
  modelLbph->train(imagesGrayscale, labels);

  while (true)
  {
    for (const auto path : validatePictures)
    {
      auto frame = cv::imread(path);
      //-- 3. Apply the classifier to the frame
      if (frame.empty())
      {
        printf(" --(!) No captured frame -- Break!");
        break;
      }

      detectAndDisplay(frame, modelLbph);
      int c = cv::waitKey(0);
      if ((char)c == 'c')
      {
        break;
      }
    }
  }

  return 0;
}

static string toString(double val)
{
  ostringstream out;
  out.precision(2);
  out << std::fixed << val;

  return out.str();
}

/** @function detectAndDisplay */
void detectAndDisplay(cv::Mat                                      frame,
                      const cv::Ptr<cv::face::LBPHFaceRecognizer>& model)
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
    auto faceGray = frame_gray(face);
    cv::resize(faceGray, faceGray, cv::Size(250, 250),
               cv::InterpolationFlags::INTER_CUBIC);

    int    predictLabel;
    double loss;
    model->predict(faceGray, predictLabel, loss);
    cv::rectangle(frame, face, CV_RGB(0, 255, 0), 1);

    const auto accuracy = loss > 100 ? 0 : 100 - loss;
    const auto message =
        predictionLabels.at(predictLabel) + "/" + toString(accuracy) + "%";
    cout << message << endl;

    cv::putText(frame, message, cv::Point(0, 20),
                cv::HersheyFonts::FONT_HERSHEY_PLAIN, 0.85,
                cv::Scalar(0, 255, 0), 2.0);
  }
  //-- Show what you got
  imshow(window_name, frame);
}