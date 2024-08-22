#include <string>
#include <chrono>

using namespace std;


typedef chrono::high_resolution_clock::time_point tim;


struct queryInfo
{
	int queryId;
	string queryString;
    tim queryRecivedTime;
    tim queryProcessedTime;


	queryInfo(){}

	queryInfo(int queryId, char* queryString)
	{
		this->queryId = queryId;
		this->queryString = queryString;
	}

    queryInfo(int queryId, char* queryString, tim queryRecivedTime)
	{
		this->queryId = queryId;
		this->queryString = queryString;
        this->queryRecivedTime = queryRecivedTime;
	}

     queryInfo(int queryId, char* queryString, tim queryRecivedTime, tim queryProcessedTime)
	{
		this->queryId = queryId;
		this->queryString = queryString;
        this->queryRecivedTime = queryRecivedTime;
        this->queryProcessedTime = queryProcessedTime;
	}
};