/* Use g++ photo_rank.cpp -lopencv_highgui -lopencv_core -lopencv_imgproc
 *
 * */

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

bool sortItems(std::pair<float, int> a, std::pair<float, int> b) {
  return a.first > b.first ? true : false;
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


int main(int argc, char **argv) {

  std::vector<std::string> img_list;

  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir("test_data")) != NULL) {
    while ((ent = readdir (dir)) != NULL) {
      img_list.push_back(std::string("test_data/") + ent->d_name);
    }
    closedir (dir);
  }
  else {
    perror ("Could not open dir.");
    return 0;
  }

  std::vector<std::pair<float, int> > ranks;
  for (int i = 0; i < img_list.size(); i++) {
    cv::Mat im;
    if (readImage(img_list[i], im) != 0) {
      img_list.erase(img_list.begin()+i);
      std::cout << "Bad image: " + img_list[i] << std::endl;
    }
    else {

      float rms = 0;
      for (int j = 1; j < 5; j++) {
        cv::Mat magI;
        sampleImage(im, magI, j);
        cv::Scalar mean = cv::mean(magI);
        rms += std::sqrt((std::pow(mean[0], 2) + std::pow(mean[1], 2) + std::pow(mean[2], 2))/3);
      }
      ranks.push_back(std::pair<float, int>(rms, i));
    }
  }

  std::sort(ranks.begin(), ranks.end(), sortItems);

  for (int i = 0; i < ranks.size(); i++) {
    cv::Mat temp;
    cv::resize(cv::imread(img_list[ranks[i].second]), temp, cv::Size(200, 200));
    cv::imshow("Rank " + std::to_string(i), temp);
    cv::waitKey(1500);
    std::cout << ranks[i].first << " " << img_list[ranks[i].second] << std::endl;
  }

  return(0);
}
