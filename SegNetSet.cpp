#include "SegNetSet.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

namespace fs = boost::filesystem;

float getMedian(const std::vector<float> &daArray) 
{
    std::vector<float> dSort(daArray);

    std::sort(dSort.begin(), dSort.end());
    // Middle or average of middle values in the sorted array.
    float dMedian = 0.0;
    size_t iSize = dSort.size();
    if ((iSize % 2) == 0) {
        dMedian = (dSort[iSize/2] + dSort[(iSize/2) - 1])/2.0;
    } else {
        dMedian = dSort[iSize/2];
    }

    return dMedian;
}

void _rotate(const cv::Mat &img, const std::string &path, 
			 const std::string &name, const int angle)
{
	cv::Mat M = cv::getRotationMatrix2D(cv::Point(img.cols / 2, img.rows / 2), angle, 1);
	cv::Mat rotM; 
	// cv::warpAffine(img, rotM, M, img.size(), cv::INTER_CUBIC, cv::BORDER_REPLICATE);
	cv::warpAffine(img, rotM, M, img.size());
	std::string _name = path + "/angle" + std::to_string(angle)+ '_' + name;

	cv::imwrite(_name.c_str(), rotM);
}

SegNetSet::SegNetSet(	const char* images, 
						const char* labels, 
						const int _num_class, 
						const int width,
						const int height):
						num_class(_num_class)
{
	d_images = new fs::path(images);
	d_labels = new fs::path(labels);
	if (!fs::exists(*d_images) || !fs::exists(*d_labels))
	{
		std::cerr << d_images << " or "<< d_labels << " does not exist\n";
		exit(1);
	}
	size = new cv::Size(width, height);
}

SegNetSet::~SegNetSet()
{
	delete d_labels;
	delete d_images;
	delete size;
}

void SegNetSet::binarization()
{
	const int step = 1;
	
	int pixel = 0;

	try
	{
		#pragma omp parallel
		#pragma omp single
        for (fs::directory_entry& file : fs::recursive_directory_iterator(*d_labels))
        {
        	#pragma omp task firstprivate(file)
        	if (fs::is_directory(file))
        	{
        		if (pixel < (num_class - 1))
        			pixel += step;

        		std::cout << file << std::endl;
        	}
        	else
        	{
        		if (file.path().filename().string()[0] != '.')
        		{
	        		cv::Mat img = cv::imread(file.path().string(), CV_LOAD_IMAGE_GRAYSCALE);
	        		img.convertTo(img, CV_8UC1);

					for (int i = 0; i < img.cols * img.rows; i++)
					{
						if (img.data[i] != 0)
						{
							img.data[i] = pixel;
						} 
					}
					cv::imwrite(file.path().string(), img);
        		}
			}
			#pragma omp taskwait
		}
	}
	catch (const fs::filesystem_error& ex)
	{
	    std::cerr << ex.what() << std::endl;
	    exit(1);
	}
}

void SegNetSet::calc_weighting()
{
	std::vector<size_t> count(num_class, 0);
	std::vector<size_t> file_count(num_class, 0); 


	unsigned int index = 0;
	bool fsize = false;
	size_t area = 0;

	std::vector<std::string> name_class(num_class - 1);

	try
	{
        for (fs::directory_entry& file : fs::recursive_directory_iterator(*d_labels))
        {
        	if (fs::is_directory(file))
        	{
        		if (index < (num_class - 1))
        		{
        			index++;
        			name_class[index - 1] = file.path().string();
        		}
        	}
        	else
        	{
        		if (file.path().filename().string()[0] != '.')
        		{
	        		cv::Mat img = cv::imread(file.path().string(), CV_LOAD_IMAGE_GRAYSCALE);
	        		img.convertTo(img, CV_8UC1);

	        		if (!fsize)
	        		{
	        			area = img.size().area();
	        			fsize = !fsize;
	        		}
	        		
	        		unsigned int curr = cv::countNonZero(img);
					
					count[index] += curr;
					count[0] += img.total() - curr;

					file_count[index]++;
					file_count[0]++;
        		}
        	}
        }
	}
	catch (const fs::filesystem_error& ex)
	{
	    std::cerr << ex.what() << std::endl;
	    exit(1);
	}

	for (size_t i = 0; i < name_class.size(); i++)
	{
		std::cout << name_class[i] << std::endl;
		std::cout << "count obj " << count[i + 1] << std::endl;
		std::cout << "file count " << file_count[i + 1] << std::endl;
	}
	std::cout << std::endl;

	std::cout << "count bg " << count[0] << std::endl;
	std::cout << "file count " << file_count[0] << std::endl;
	std::cout << std::endl;


	std::vector<float> freq(num_class, 0.0);

	for (int i = 0; i < num_class; ++i)
	{
		freq[i] = (float)count[i] / (file_count[i] * area);
	}
	

	float median = getMedian(freq);

	std::cout << "Class weighting" << std::endl;
	std::vector<float> weighting(num_class, 0.0);
	for (int i = 0; i < num_class; ++i)
	{
		weighting[i] = median / freq[i];
		std::cout << weighting[i] << std::endl;
	}
}

void SegNetSet::rotate(const fs::path* dir)
{
	try
	{
		#pragma omp parallel
		#pragma omp single
        for (fs::directory_entry& file : fs::recursive_directory_iterator(*dir))
        {
        	#pragma omp task firstprivate(file)
        	if (fs::is_regular_file(file))
        	{
        		const std::string name =  file.path().filename().string();
        		std::size_t found = name.find("angle");
        		if ((found == std::string::npos) && (name[0] != '.'))
        		{
        			const std::string parent = file.path().parent_path().string();
	        		// cv::Mat img = cv::imread(file.path().string(), CV_LOAD_IMAGE_GRAYSCALE);
	        		cv::Mat img = cv::imread(file.path().string());

	        		_rotate(img, parent, name, 30);
	        		_rotate(img, parent, name, -30);

	        		_rotate(img, parent, name, 60);
	        		_rotate(img, parent, name, -60);
        		}
	        }
	        #pragma omp taskwait
        }
	}
	catch (const fs::filesystem_error& ex)
	{
	    std::cerr << ex.what() << std::endl;
	    exit(1);
	}

}

void SegNetSet::crop()
{
	std::string list = d_images->parent_path().parent_path().string() + "/" + "list.txt";
	std::fstream fs (list, std::fstream::in);

	if(!fs.is_open())
	{
		std::cout << "Error open list of images and labels!" << std::endl;
		std::cout << "Create list ..." << std::endl; 
		fs.close();
		make_list();
		fs.open(list);
		std::cout << "Create list successful" << std::endl;
	}

	//Rand width and height
	std::random_device random_device; // Источник энтропии.
	std::mt19937 generator(random_device()); // Генератор случайных чисел.
	// (Здесь берется одно инициализирующее значение, можно брать больше)
	std::uniform_int_distribution<> width_distribution(400, 500);
	std::uniform_int_distribution<> height_distribution(350, 500); 

	std::string name_i, name_l;
	std::string parent_img = d_images->parent_path().parent_path().string();
	std::string parent_lab = d_labels->parent_path().parent_path().string();
	while(!std::getline(fs, name_i, ' ').eof())
	{
		std::getline(fs, name_l);

		std::string path_img = parent_img + "/" + name_i;
		std::string path_lab = parent_lab + "/" + name_l;

		cv::Mat img;
		cv::Mat lab;
		try 
		{
		    img = cv::imread(path_img, CV_LOAD_IMAGE_GRAYSCALE);
		    lab = cv::imread(path_lab, CV_LOAD_IMAGE_GRAYSCALE);
		}
		catch (cv::Exception& ex) 
		{
		    std::cerr << "Exception open image: " << ex.what() << std::endl;
		    exit(1);
		}
		

		float rand_width = width_distribution(generator);
		float rand_height = height_distribution(generator);
		
		cv::Mat _img = img(cv::Rect(img.cols / 2 - rand_width / 2, 
		                            img.rows / 2.2 - rand_height / 2,
		                            rand_width, rand_height));
		cv::Mat _lab = lab(cv::Rect(lab.cols / 2 - rand_width / 2, 
		                            lab.rows / 2.2 - rand_height / 2,
		                            rand_width, rand_height));

		cv::imwrite(path_img, _img);
		cv::imwrite(path_lab, _lab);
	}
   
	fs.close();
}

void SegNetSet::make_list()
{
	std::vector<std::pair<std::string, std::string>> img;
	std::vector<std::pair<std::string, std::string>> lab;
	try
	{
	    for (const fs::directory_entry& file : fs::recursive_directory_iterator(*d_images))
	    {
	    	const std::string name = file.path().filename().string(); 
	    	if (fs::is_regular_file(file) && (name[0] != '.'))
	    	{
	    		std::string parent = file.path().parent_path().filename().string();
	    		img.push_back(std::make_pair(parent, name));
	        }
	    }

	    for (const fs::directory_entry& file : fs::recursive_directory_iterator(*d_labels))
	    {
	    	const std::string name = file.path().filename().string(); 
	    	if (fs::is_regular_file(file) && (name[0] != '.'))
	    	{
	    		std::string parent = file.path().parent_path().filename().string();
	    		lab.push_back(std::make_pair(parent, name));
	        }
	    }
	}
	catch (const fs::filesystem_error& ex)
	{
	    std::cerr << ex.what() << std::endl;
	    exit(1);
	}

	std::sort(img.begin(), img.end(), [](std::pair<std::string, std::string> a, std::pair<std::string, std::string> b){
		return a.second < b.second;
	});
	std::sort(lab.begin(), lab.end(), [](std::pair<std::string, std::string> a, std::pair<std::string, std::string> b){
		return a.second < b.second;
	});

	std::string path_list = d_images->parent_path().parent_path().string() + "/" + "list.txt";
	std::cout << path_list << std::endl;
	std::fstream fs (path_list, std::fstream::out | std::fstream::trunc);

	for (auto i = img.begin(), j = lab.begin(); i != img.end() && j != lab.end(); ++i, ++j)
	{
		fs << i->first + "/" + i->second << " ";
		fs << "labels/" + j->first + "/" + j->second << std::endl;
	}

	fs.close();
}

void SegNetSet::resize(const fs::path* dir)
{
	try
	{
		#pragma omp parallel
		#pragma omp single
        for (fs::directory_entry& file : fs::recursive_directory_iterator(*dir))
        {
        	#pragma omp task firstprivate(file)
        	if (fs::is_regular_file(file))
        	{
        		const std::string name =  file.path().filename().string();
        		if (name[0] != '.')
        		{
	        		cv::Mat img = cv::imread(file.path().string(), CV_LOAD_IMAGE_GRAYSCALE);

	        		cv::Mat _img;
	        		cv::resize(img, _img, *size, 0, 0, cv::INTER_NEAREST);

	        		imwrite(file.path().string(), _img);
        		}
	        }
	        #pragma omp taskwait
        }
	}
	catch (const fs::filesystem_error& ex)
	{
	    std::cerr << ex.what() << std::endl;
	    exit(1);
	}
}

void SegNetSet::Prepare(const Flags *flags)
{
	std::cout << "Start prepare sets for SegNet" << std::endl;
	std::cout << std::endl;

	if (flags->rotate)
	{
		// std::cout << "Rotate images ..." << std::endl;
		// this->rotate(d_images);
		// std::cout << "Rotate images finished successfully" << std::endl;
		
		std::cout << std::endl;

		std::cout << "Rotate labels ..." << std::endl;
		this->rotate(d_labels);
		std::cout << "Rotate labels finished successfully" << std::endl;
	}

	if (flags->list)
	{
		std::cout << "Create list ..." << std::endl;
		this->make_list();
		std::cout << "Create list successful" << std::endl;
	}

	if (flags->crop)
	{
		std::cout << "Crop imgaes and labels ..." << std::endl;
		this->crop();
		std::cout << "Crop finished successfully" << std::endl;
	}

	if (flags->bin)
	{
		std::cout << "Transform labels ..." << std::endl;
		this->binarization();
		std::cout << "Transform labels finished successfully" << std::endl;
	}

	if (flags->weights)
	{
		std::cout << "Calculate weightings ..." << std::endl;
		std::cout << std::endl;
		this->calc_weighting();
		std::cout << std::endl;
		std::cout << "Calculate successful" << std::endl;
	}

}