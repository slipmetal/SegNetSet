#include <iostream>
#include <string>

#include "SegNetSet.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

int main(int argc, char const *argv[])
{
    Flags flags;
    int num = 0;

    std::string path_img;
    std::string path_lab;
    std::vector<int> size(2);
	try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "Produce help message")
            ("images,i",    po::value<std::string>(&path_img),  "Images path")
            ("labels,l",    po::value<std::string>(&path_lab),  "Labels path")
            ("num,n",       po::value<int>(&num),               "Number of classes")
            ("all,a",                                           "All stages prepare")      
            ("rotate,r",                                        "Rotate images and labels")
            ("bin,b",                                           "Label binarization")
            ("weights,w",                                       "Calculation of weights")
            ("list,m",                                          "Create list of images and labels")
            ("crop,c",                                          "Random crop imgaes and labels")
            ("size,s",      po::value<std::vector<int>>(&size)->multitoken(), "Resize imgaes and labels")
        ;

        po::variables_map vm;        
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }

        if (vm.count("images"))
        {
            std::cout << "Images path was set to " << path_img << "\n";
        }
        else
        {
            std::cerr << "Images path was not set" << "\n";
            return 1;
        }

        if (vm.count("labels"))
        {
            std::cout << "Labels path was set to " << path_lab << "\n";
        }
        else
        {
            std::cerr << "Labels path was not set" << "\n";
            return 1;
        }


        if (vm.count("num")) 
        {
            std::cout << "Number of classes was set to " 
                 << num << ".\n";
        } 
        else 
        {
            std::cout << "Number of classes was not set.\n";
            return 1;
        }

        if (vm.count("rotate"))
        {
            flags.rotate = true;
            std::cout << "Rotate images and labels" << ".\n";   
        }
        if (vm.count("bin"))
        {
            flags.bin = true;
            std::cout << "Flag label binarization was set to" << ".\n";
        }

        if (vm.count("weights"))
        {
            flags.weights = true;
            std::cout << "Calculation of weights" << ".\n";
        }


        if (vm.count("list"))
        {
            flags.list = true;
            std::cout << "Create list of images and labels" << ".\n";
        }

        if (vm.count("crop"))
        {
            flags.crop = true;
            std::cout << "Crop imgaes and labels" << ".\n";
        }

        if (vm.count("size"))
        {
            flags.size = true;
            std::cout << "Images and labels resize to " << size[0] << "x" << size[1] << ".\n";
        }

        if (vm.count("all"))
        {
            flags.crop = true;
            flags.list = true;
            flags.weights = true;
            flags.bin = true;
            flags.rotate = true;
            flags.size = true;
            std::cout << "All steps" << ".\n";
        }
    }
    catch(std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        std::cerr << "Exception of unknown type!\n";
    }


    SegNetSet set(path_img.c_str(), path_lab.c_str(), num, size[0], size[1]);

    set.Prepare(&flags);

	return 0;
}