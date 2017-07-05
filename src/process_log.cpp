//The following code is a solution to the challenge presented by Isight Data Science:
// https://github.com/InsightDataScience/anomaly_detection
//The full problem statement is decribed at the above URL, I will not repeat it here.

//The solution presented here builds an adjancency matrix from the initial "batch" input.
//The (sparse and symmetric) adjacency matrix has value True at element (i,j) when customers 
// i and j are friends, and False if they are not friends.
//To determine the social network of depth D of customer i, we can take the ith row of the 
//  adjacency matrix and multiply it by the full adjacency matrix D-1 times

//The purchase history of each customer is stored as a deque, which is a like a queue
//but with random element access.
//The purchase histories of all customers are stored as a vector<deque>


//Assumptions: 
// 1) this solution is only efficient if the adjacency matrix is indeed sparse.
//		Otherwise, the solution will still work but will be slower than dense matrix representation
// 2) All possible unique users are listed in the input batch file. 
//		E.g. if the batch file has ids in the range [0,N] then the sample test will also have 
//		ids in the range [0,N]
// 3) Input data is pre-sorted in chronological order

#include<iostream>
#include<fstream>
#include<math.h>
#include<vector>
#include<string>
#include<sstream>
#include<ctime>
#include<deque>
#include<algorithm>
#include "json.hpp"//this (very) simple library allows me to read json elements in C++
					//The library is far from the fastest/best, but is the easiest to implement
					//This code can be sped up (at the cost of installation ease) by using 
					// a better json library. 
#include<Eigen/Sparse> //Eigen is a matrix manipulation library. It's basically a C++ wrapper for LAPACK/BLAS
#include<unsupported/Eigen/MatrixFunctions>


using json = nlohmann::json; 


//The chronological order of purchases is important
//This struct bundles together the amount of the purchase (dollars)
//with its timestamp and with the order in which this purchase came in the stream
//(the latter is important for comparing purchases made with the same timestamp)
struct purchase{
	double amount;
	time_t t;
	unsigned int pos;
};

//To sort purchases we need to sort primarily by timestamp, secondarily by position in the stream.
struct compare_purchase{
	inline bool operator() (const purchase& p1, const purchase& p2)
	{
		if (p1.t != p2.t)
			return (p1.t < p2.t);
		else 
			return (p1.pos < p2.pos);
	}
};


//This method resizes the two adjacency matrices and the purchase history vector
//Note: it could be worth exploring the possibility of optimizing this by resizing to 
//  a size LARGER than new_size. If I find that I resizing often then it's better to 
//	just resize to a much larger size ONCE, rather than resize a little bit a bunch of times.
void resize(Eigen::SparseMatrix<bool,Eigen::RowMajor> *adMat, std::vector<std::deque<purchase>*>* purchase_history, int new_size,int *num_customers)
{
	adMat->resize(new_size+1,new_size+1); 
	for (int i=purchase_history->size();i<new_size+1;i++)
		adMat->coeffRef(i,i)=1;
	
	for (int i=purchase_history->size(); i<new_size+1;i++)
		purchase_history->push_back(new std::deque<purchase>());

	*num_customers = new_size+1;
}


//This method determines whether a particular purchase was anomolous
//Input: 
//  vec: a vector of pointers, each pointer points to the beginning of a queue containing the previous
//  T purchases of one of the members in this customer's social network
//  price: the cost of the current purchase
//  mean, std: pointers to where the computed mean/std should be stored
//  T: number of purchases that should be considered for a particular mean/std calculation
//Output:
//	a boolean representing whether this purchase is anomolous. 
bool isAnomalous(std::vector<std::deque<purchase>*> *vec, double price, double* mean, double* std, int T)
{

	//first, copy all of the neighbors' purchasing history into one giant queue:
	//NOTE: This is a rather inefficient (but quick to implement) way to do this!
	//  The better way is to /not/ copy all of the data into one queue, but instead
	//	have an array of pointers to the current position in each queue
	//	and at each step determine the earliest purchase pointed to by these pointers.
	//	Then, remember the position of this pointer, and increment it.
	//  Keep doing this until each pointer reaches the end of the queue OR you've already
	//	seen T different purchases.
	//	Since the size of the neighborhood isn't that large and the number of
	//  remembered purchases is not large, I think that this improvement isnt worth the time needed to implemenet.
	std::deque<purchase> combo;
	for (int i=0;i<vec->size();i++)
		for (int j=0;j<vec->at(i)->size();j++)
		{
			purchase p;
			p.amount = vec->at(i)->at(j).amount;
			p.t = vec->at(i)->at(j).t;
			p.pos = vec->at(i)->at(j).pos;
			combo.push_back(p);
		}

	//now sort the queue
	//I wrote my own comparison operator that sorts primarily by timestamp, secondarily by position in stream
	std::sort(combo.begin(),combo.end(),compare_purchase());

	//check to make sure the number of data points is >2:
	*mean = 0.;
	*std = 0.;
	if (combo.size() <= 2)
	{
		*mean = -1.;
		*std = -1.;
		return false;
	}

	//now compute the average and standard deviation
	int limit = T;
	if (combo.size() < T)
		limit = combo.size();

	for (int i=0;i < limit; i++)
		*mean += combo.at(i).amount;
	*mean /= limit;
	for (int i=0;i < limit; i++)
		*std += pow(combo.at(i).amount-*mean,2);
	*std /= limit;
	*std = sqrt(*std);


	if (fabs(price - *mean) > 3* *std)
		return true;
	return false;
}




//Like a true scientist, the bulk of my code is in main()...
int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		std::cout << "need 3 command line arguments" << std::endl;
		std::cerr << "ERROR: need 3 arguments" << std::endl;
		exit(1);
	}

	std::ifstream batch(argv[1]);
	if (!batch.good())
	{
		std::cout << "Failed. See error log for details" << std::endl;
		std::cerr << "ERROR: first argument (batch file) not readable" << std::endl;
		exit(1);
	}

	std::ifstream stream(argv[2]);
	if (!batch.good())
	{
		std::cout << "Failed. See error log for details" << std::endl;
		std::cerr << "ERROR: second argument (stream file) not readable" << std::endl;
		exit(1);
	}

	FILE *output;
	output = fopen(argv[3],"w");
	if (output == 0)
	{
		std::cout << "Failed. See error log for details" << std::endl;
		std::cerr << "ERROR: third argument (output file) not writable" << std::endl;
		exit(1);
	}


	std::stringstream ss;  //stringstream for converting json input to c-variables

	//First, read the parameters T and D:
	json params;
	batch >> params;
	
	ss.str(std::string()); ss.clear();
	ss << params["D"];
	int D = atoi(ss.str().substr(1,ss.str().size()-2).c_str()); //convert "##" to int

	ss.str(std::string()); ss.clear();
	ss << params["T"];
	int T = atoi(ss.str().substr(1,ss.str().size()-2).c_str()); //convert "##" to int
	

	//Now, initialize the adjacency matrix and its exponent M^D
	//The initial size of the matrix is small, but it will be expanded 
	//	if the number of customers is larger than 10
	int num_customers = 10;	
	Eigen::SparseMatrix<bool,Eigen::RowMajor> adMat(num_customers, num_customers); 

	//populate the diagonals of the adjacency matrix:
	for (int i=0;i<num_customers;i++)
		adMat.coeffRef(i,i)=1;

	//Declare some useful variables:
	int itmp; //variable for short-term integer storage
	double dtmp; //variable for short-term double storage
	
	//some variables for dealing with time:
	struct std::tm tm;
	time_t t;
	time_t prev_time = 0;
	unsigned int num_time = 0; //counts the position of the entry in the file 


	//initialize a vector of queues
	//each queue corresponds to the previous T purchases of a customer
	std::vector<std::deque<purchase>*> purchase_history(num_customers);
	for (int i=0;i<num_customers;i++)
		purchase_history.at(i) = new std::deque<purchase>();


/* ------------------------Now we read the batch file --------------------------*/


	//first we must read the batch file, check the data for errors, 
	// reformat the data (cast the inputs as the appropriate data types),
	// build the adjacency matrix, and save all the users buying history
	int count =0;
	std::cout << "READING BATCH FILE" << std::endl;
	while (!batch.eof())
	{	
		json log;

		//try to read a json element from the input file:
		try{
			batch >> log;
		}
		catch (int e) {std::cerr << "Caught exception  when reading batch jsons. Code" << e;} 
		catch (char e) {std::cerr << "Caught exception  when reading batch jsons. Code" << e;} 
		catch (std::string e) {std::cerr << "Caught exception  when reading batch jsons. Code" << e;} 
		catch (...) {std::cerr << "Caught default exception when reading stream jsons." << std::endl;break;}

		//convert timestamp to a time:
		ss.str(std::string()); ss.clear();
		ss << log["timestamp"];
		strptime(ss.str().c_str(),"\"%Y-%m-%d %H:%M:%S\"",&tm); //this is POSIX standard method
										//which *may* not be available to Windows.... should be okay though
		std::time_t time = mktime(&tm);
		log["timestamp"] = time; //time is seconds since Jan 1 1970 (POSIX standard)
		if (prev_time != time)
		{
			prev_time = time;
			num_time = 0;
		}
		else num_time++;

		//get the event type:
		ss.str( std::string() ); ss.clear();
		ss << log["event_type"];			
		if (std::string("\"purchase\"").compare(ss.str())==0)
		{
			//convert ID to int and amount to double:
			ss.str(std::string()); ss.clear();
			ss << log["id"];
			itmp = atoi(ss.str().substr(1,ss.str().size()-2).c_str()); //convert id to int
			log["id"] = itmp;
			ss.str( std::string() ); ss.clear();
			ss << log["amount"];
			dtmp = atof(ss.str().substr(1,ss.str().size()-2).c_str()); //convert amount to double
			log["amount"] = dtmp;
			
			//check if new ID falls outside of matrix bounds
			//if it does, resize the matrices and purchase_history vectors
			if (itmp >= num_customers)
				resize(&adMat,&purchase_history,itmp,&num_customers);
			
			//store purchase information in a purchase structure:
			purchase cur;
			cur.amount = dtmp;
			cur.t = time;
			cur.pos = num_time;
	
			//add this purchase to the corresponding history queue
			purchase_history.at(itmp)->push_back(cur);
			if (purchase_history.at(itmp)->size() > T)
				purchase_history.at(itmp)->pop_front();
		}
		else //befriend or unfriend
		{
			int id1,id2;
			//convert both IDs to ints:
			ss.str(std::string()); ss.clear();
			ss << log["id1"];
			id1 = atoi(ss.str().substr(1,ss.str().size()-2).c_str()); //convert id1 to int
			log["id1"] = itmp;
			ss.str(std::string()); ss.clear();
			ss << log["id2"]; 
			id2 = atoi(ss.str().substr(1,ss.str().size()-2).c_str()); //convert id2 to int
			log["id2"] = itmp;

			//check if new IDs fall outside of matrix bounds
			//if they do, resize the matrices and purchase_history vectors
			if (std::max(id1,id2) >= num_customers)
				resize(&adMat,&purchase_history,std::max(id1,id2),&num_customers);


			ss.str(std::string()); ss.clear();
			ss << log["event_type"];
			if (std::string("\"befriend\"").compare(ss.str())==0)
			{
				//add friendship to adjacency matrix
				adMat.coeffRef(id1,id2)=1;
				adMat.coeffRef(id2,id1)=1;
			}
			else  //unfriend
			{
				//remove friendship from adjancency matrix
				adMat.coeffRef(id1,id2)=0;
				adMat.coeffRef(id2,id1)=0;
			}
		}

	}
	batch.close();

	std::cout << "...DONE" << std::endl;
/*

--------------------------------------------------------------------------------

At this point we have read all the data from the batch file
We have a vector of all the previous purchases and an adjacency matrix 
that describes who is friends with who to degree D

The next step is to read the stream file, work up the json entries in the 
exact same way, but this time check for anomalous purchasing behavior

--------------------------------------------------------------------------------


*/	

	std::cout << "READING STREAM FILE" << std::endl;
	double mean;
	double std;
	//ASIDE: You may notice that the code below is repetitive with the code above. 
	//	I am aware that this is bad practice and I should modularize this json workup code,
	//	However, I'm short on time and there are bigger/more important issues with the code
	while (!stream.eof()) 
	{	
		json log;

		//try to read a json element from the input file:
		try{
			stream >> log;
		}
		catch (int e) {std::cerr << "Caught exception  when reading stream jsons. Code " << e;} 
		catch (char e) {std::cerr << "Caught exception  when reading stream jsons. Code " << e;} 
		catch (std::string e) {std::cerr << "Caught exception  when reading stream jsons. Code" << e;} 
		catch (...) {std::cerr << "Caught default exception when reading stream jsons." << std::endl;break;}

		//convert timestamp to a time:
		ss.str(std::string()); ss.clear();
		ss << log["timestamp"];
		strptime(ss.str().c_str(),"\"%Y-%m-%d %H:%M:%S\"",&tm); //this is POSIX standard method
										//which *may* not be available to Windows....
		std::time_t time = mktime(&tm);
		log["timestamp"] = time; //time is seconds since Jan 1 1970 (POSIX standard)
		if (prev_time != time)
		{
			if (prev_time > time)
			{
				std::cerr << "ERROR: BATCH FILE NOT CHRONOLOGICALLY SORTED" << std::endl;
				std::cout << "Unsucessful. See error log (default ./log_output/error_log" << std::endl;
				exit(2);
			}
			prev_time = time;
			num_time = 0;
		}
		else num_time++;


		ss.str( std::string() ); ss.clear();
		ss << log["event_type"];	
		if (std::string("\"purchase\"").compare(ss.str())==0)
		{
			//convert ID to int and amount to double:
			ss.str(std::string()); ss.clear();
			ss << log["id"];
			itmp = atoi(ss.str().substr(1,ss.str().size()-2).c_str()); //convert id to int
			log["id"] = itmp;
			ss.str( std::string() ); ss.clear();
			ss << log["amount"];
			dtmp = atof(ss.str().substr(1,ss.str().size()-2).c_str()); //convert amount to double
			log["amount"] = dtmp;
			
			//add this purchase to the queue of all of this user's friends:
			purchase cur; 
			cur.amount = dtmp;
			cur.t = time;
			cur.pos = num_time;


			//The next few lines are the crux of this code ----- 
			//We will compute this particular user's social network and determine whether his/her purchase is anomalous

			//for convenience, give this particular user a name: Sally
			//Now we must determine who is in Sally's social network, and whether her purchase is anomalous
			std::vector<std::deque<purchase>*> network_purchases; //single out the purchases of Sally's network
			Eigen::SparseMatrix<bool, Eigen::RowMajor> neighbors=adMat.row(log["id"].get<int>()); //These are Sally's friends
			for (int i=0; i<D-1; i++)
				neighbors = neighbors * adMat; //each multiplication by adMat gives another layer of Sally's network

			//At this point, neighbors stores a vector<bool> that is true for Sally's friends of degree D 
			for (int i=0;i<num_customers;i++)
			{
				if (i==log["id"]) continue; //exclude Sally her own network
				if (neighbors.coeffRef(0,i)) 
					network_purchases.push_back(purchase_history.at(i)); 
			}
			
			//test whether this purchase is anomalous
			if (isAnomalous(&network_purchases,dtmp,&mean, &std, T))
			{
				// if it is anomalous, print the log to the output file in json format
				char timestr[100];
				unsigned int timestamp = log["timestamp"].get<unsigned int>();
				std::time_t rawtime = timestamp;

				struct tm *dt;
				dt = localtime(&rawtime);
				std::strftime(timestr, sizeof(timestr),"\"%Y-%m-%d %H:%M:%S\"",dt);
				std::stringstream tmp_ss;
				tmp_ss << "{\"event_type\":" << log["event_type"]<<", \"timestamp\":"<<timestr;
				tmp_ss << ", \"id\": \"" << log["id"] << "\", \"amount\": \"";
				fprintf(output,tmp_ss.str().c_str());
				fprintf(output,"%.2f",log["amount"].get<double>());
  				fprintf(output,"\", \"mean\": \"");
				fprintf(output,"%.2f",mean);
  				fprintf(output,"\", \"sd\": \"");
				fprintf(output,"%.2f",std);
				fprintf(output,"\"}\n");
			}

			//add the purchase to the queue
			purchase_history.at(itmp)->push_back(cur);
			if (purchase_history.at(itmp)->size() > T)
				purchase_history.at(itmp)->pop_front();

		}
		else //befriend or unfriend
		{
			int id1,id2;
			//convert both IDs to ints:
			ss.str(std::string()); ss.clear();
			ss << log["id1"];
			id1 = atoi(ss.str().substr(1,ss.str().size()-2).c_str()); //convert id1 to int
			log["id1"] = itmp;
			ss.str(std::string()); ss.clear();
			ss << log["id2"]; 
			id2 = atoi(ss.str().substr(1,ss.str().size()-2).c_str()); //convert id2 to int
			log["id2"] = itmp;

			//check if new IDs fall outside of matrix bounds
			//if they do, resize the matrices and purchase_history vectors
			if (std::max(id1,id2) >= num_customers)
				resize(&adMat,&purchase_history,std::max(id1,id2),&num_customers);


			ss.str(std::string()); ss.clear();
			ss << log["event_type"];
			if (std::string("\"befriend\"").compare(ss.str())==0)
			{
				//add friendship to adjacency matrix
				adMat.coeffRef(id1,id2)=1;
				adMat.coeffRef(id2,id1)=1;
			}
			else  //unfriend
			{
				//remove friendship from adjancency matrix
				adMat.coeffRef(id1,id2)=0;
				adMat.coeffRef(id2,id1)=0;
			}
		}
	}

	std::cout << "...DONE" << std::endl;
	for (int i=0;i<num_customers;i++)
		delete purchase_history.at(i); 
}


/*
old code for the stream workup
			//convert both IDs to ints:
			ss.str(std::string()); ss.clear();
			ss << log["id1"];
			itmp = atoi(ss.str().substr(1,ss.str().size()-2).c_str()); //convert id1 to int
			log["id1"] = itmp;
			ss.str(std::string()); ss.clear();
			ss << log["id2"]; 
			itmp = atoi(ss.str().substr(1,ss.str().size()-2).c_str()); //convert id2 to int
			log["id2"] = itmp;
			ss.str(std::string()); ss.clear();


			ss << log["event_type"];
			if (std::string("\"befriend\"").compare(ss.str())==0)
			{
				//add friendship to adjacency matrix
				adMat.coeffRef(log["id1"].get<int>(),log["id2"].get<int>())=1;
				adMat.coeffRef(log["id2"].get<int>(),log["id1"].get<int>())=1;
			}
			else  //unfriend
			{
				//remove friendship from adjancency matrix
				adMat.coeffRef(log["id1"].get<int>(),log["id2"].get<int>())=0;
				adMat.coeffRef(log["id2"].get<int>(),log["id1"].get<int>())=0;
			}
*/
