#include "../ipc/msgtool.h"
#include <iostream>
using namespace std;

int main(int argc, char *argv[])
{
    msg_queue_client client('g');
    long type = 1;
    char tmp[1000];

    while(true)
    {
        cout<<"Input Query: ";
        cin.get(tmp, 1000);//readLine
        cin.get();//skip line-breaker
        if(strcmp(tmp, "exit")==0) return 0;
        client.send_msg(type, tmp);
    }

    return 0;
}