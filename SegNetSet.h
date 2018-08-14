#ifndef SEGNET_SET_H_
#define SEGNET_SET_H_

#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>

struct Flags
{
	bool rotate;                                    
    bool bin;                                           
    bool weights;                                       
    bool list;                                          
    bool crop;                                          
    bool size;

    Flags(): rotate(false), bin(false), weights(false), list(false), crop(false), size(false)
    {}      
};

class SegNetSet
{
public:
	SegNetSet(	const char* images, const char* labels, 
				const int _num_class, const int width,
				const int height);
	~SegNetSet();
	void Prepare(const Flags *flags);

private:
	boost::filesystem::path *d_images;
	boost::filesystem::path *d_labels;
	cv::Size *size;
	int num_class;

	void binarization();
	void calc_weighting();
	void rotate(const boost::filesystem::path *dir);
	void make_list();
	void crop();
	void resize(const boost::filesystem::path *dir);

};
#endif // SEGNET_SETS_H_