#include <iostream>
#include <string>
#include <cmath>
#include <vector>

#include "dirent.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define CENTRE 1
#define RIGHT 2
#define LEFT 3
#define UP 4
#define DOWN 5

bool sortItems(std::pair<std::string, double> a, std::pair<std::string, double> b) {
  return a.second > b.second ? true : false;
}

int readImage(std::string filepath, cv::Mat &normIm) {

  cv::Mat im = cv::imread(filepath, CV_LOAD_IMAGE_GRAYSCALE);
  if (im.empty()) {
    return -1;
  }

  cv::resize(im, normIm, cv::Size(200, 200));

  return 0;
}

int computeDFT(const cv::Mat &im, cv::Mat &magI) {

  cv::Mat padded;
  int m = cv::getOptimalDFTSize(im.rows);
  int n = cv::getOptimalDFTSize(im.cols);
  copyMakeBorder(im, padded, 0, m - im.rows, 0, n - im.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

  cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
  cv::Mat complexI;
  merge(planes, 2, complexI);

  cv::dft(complexI, complexI);

  cv::split(complexI, planes);
  cv::magnitude(planes[0], planes[1], planes[0]);
  magI = planes[0];

  magI += cv::Scalar::all(1);
  log(magI, magI);

  magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

  int cx = magI.cols/2;
  int cy = magI.rows/2;

  cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));
  cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));
  cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));
  cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy));

  cv::Mat temp;
  q0.copyTo(temp);
  q3.copyTo(q0);
  temp.copyTo(q3);

  q1.copyTo(temp);
  q2.copyTo(q1);
  temp.copyTo(q2);

  cv::normalize(magI, magI, 0, 1, CV_MINMAX);

  return 0;
}

int sampleImage(const cv::Mat &im, cv::Mat &spectrum, int pos) {

  int sx, sy, ex, ey;

  if (pos ==  CENTRE) {
    sx = im.cols/4;
    sy = im.rows/4;
    ex = im.cols/2 + im.cols/4;
    ey = im.rows/2 + im.rows/4;
  }
  else if (pos ==  LEFT) {
    sx = 0;
    sy = 0;
    ex = im.cols/4;
    ey = im.rows;
  }
  else if (pos ==  RIGHT) {
    sx = im.cols/2 + im.cols/4;
    sy = 0;
    ex = im.cols;
    ey = im.rows;
  }
  else if (pos ==  UP) {
    sx = 0;
    sy = 0;
    ex = im.cols;
    ey = im.rows/4;
  }
  else if (pos ==  DOWN) {
    sx = im.cols/2 + im.cols/4;
    sy = im.rows/2 + im.rows/4;
    ex = im.cols;
    ey = im.rows;
  }

  cv::Mat patch;
  cv::resize(cv::Mat(im, cv::Rect(sx, sy, ex-sx, ey-sy)), patch, cv::Size(200, 200));
  computeDFT(patch, spectrum);

  return 0;
}

std::vector<std::string>* getImageFileNamesList(std::string imageRepositoryLocation) {
  std::vector<std::string> *img_list = new std::vector<std::string>();

  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir(imageRepositoryLocation.c_str())) != NULL) {
    while ((ent = readdir (dir)) != NULL) {
      if (std::string(ent->d_name).find(".jpg") != std::string::npos) {
        img_list->push_back(imageRepositoryLocation + "/" + ent->d_name);
      }
    }
    closedir (dir);
  }
  else {
    perror ("Could not open dir.");
    return 0;
  }

  return img_list;
}

double rankImage(std::string imageFilename) {
  cv::Mat im;
  double rms = 0;
  if (readImage(imageFilename, im) == 0) {
    for (int j = 1; j < 5; j++) {
      cv::Mat magI;
      sampleImage(im, magI, j);
      cv::Scalar mean = cv::mean(magI);
      rms += std::sqrt((std::pow(mean[0], 2) + std::pow(mean[1], 2) + std::pow(mean[2], 2))/3);
    }
  }

  return rms;
}

void showImages(std::vector<std::pair<std::string, double> > rankedImageList) {
  for (std::pair<std::string, double> rankImagePair : rankedImageList) {
    cv::Mat image;
    cv::resize(cv::imread(rankImagePair.first), image, cv::Size(200, 200));
    cv::imshow(rankImagePair.first, image);
    cv::waitKey(1500);
  }
}


int main(int argc, char **argv) {

  std::string imageRepositoryLocation = "data/test_data";

  std::vector<std::string> *imageFilenameList = getImageFileNamesList(imageRepositoryLocation);

  std::vector<std::pair<std::string, double> > rankedImageList;

  for (std::string imageFileName : *imageFilenameList) {
    double imageRankScore = rankImage(imageFileName);
    rankedImageList.push_back(std::pair<std::string, double>(imageFileName, imageRankScore));
  }

  std::sort(rankedImageList.begin(), rankedImageList.end(), sortItems);

  for (std::pair<std::string, double> rankImagePair : rankedImageList) {
    std::cout << rankImagePair.first << " " << rankImagePair.second << std::endl;
  }

  showImages(rankedImageList);

  return(0);
}
