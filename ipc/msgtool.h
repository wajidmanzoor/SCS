#pragma once
//This is a wrapper of IPC message buffer

//If eclipse reports "__gxx_personality_v0" problem, following the instruction below to set eclipse
//http://omtlab.com/configure-your-eclipse-for-c/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <errno.h>

#define MAX_SEND_SIZE 80
#define MSG_DIR "/tmp"

//----------------------------------------------------------------------------

struct string_msg_buf {
	long mtype;//type=0 means get all kinds of msgs, otherwise type>0
	char mtext[MAX_SEND_SIZE];
};

//----------------------------------------------------------------------------

struct msg_queue_server
{
	char service_num;
	key_t key;//queue key
	int qid;//queue id
	string_msg_buf msg;//for holding the current msg

	/*msg_queue_server()
	{
		service_num = 'a';
		key = ftok(MSG_DIR, service_num);
		if((qid = msgget(key, IPC_CREAT|IPC_EXCL|0660)) == -1) {
			perror("Error in msg_queue_server's constructor");
			exit(1);
		}
	}*/

    msg_queue_server() {
    service_num = 'a';
    key = ftok(MSG_DIR, service_num);
    if ((qid = msgget(key, IPC_CREAT | 0660)) == -1) {
        if (errno == EEXIST) {
            // The queue already exists, so just open it
            if ((qid = msgget(key, 0)) == -1) {
                perror("Error in msg_queue_server's constructor (opening existing queue)");
                exit(1);
            }
        } else {
            perror("Error in msg_queue_server's constructor (creating new queue)");
            exit(1);
        	}
    	}
	}

	msg_queue_server(char serviceNo) {
    service_num = serviceNo;
    key = ftok(MSG_DIR, service_num);
    if ((qid = msgget(key, IPC_CREAT | 0660)) == -1) {
        if (errno == EEXIST) {
            // The queue already exists, so just open it
            if ((qid = msgget(key, 0)) == -1) {
                perror("Error in msg_queue_server's constructor (opening existing queue)");
                exit(1);
            }
        } else {
            perror("Error in msg_queue_server's constructor (creating new queue)");
            exit(1);
        	}
    	}
	}

	/*msg_queue_server(char serviceNo)
	{
		service_num = serviceNo;
		key = ftok(MSG_DIR, service_num);
		if((qid = msgget(key, IPC_CREAT|IPC_EXCL|0660)) == -1) {
			perror("Error in msg_queue_server's constructor");
			exit(1);
		}
	}*/

	//first recv_msg(), then get_msg() if received

	bool recv_msg(long type)
	{
		//true: msg received, call get_msg() to get the msg
		//false: no msg available
		msg.mtype = type;
		if((msgrcv(qid, (msgbuf *)&msg,
				MAX_SEND_SIZE, type, IPC_NOWAIT))==-1)
		{
			if (errno != ENOMSG)
			{
				perror("Error in msg_queue_server's recv_message()");
				exit(1);
			}
			return false;
		}
		return true;
	}

	char* get_msg()
	{
		return msg.mtext;
	}

	~msg_queue_server()
	{
		if(msgctl(qid, IPC_RMID, 0) == -1)
		{
			perror("Error in msg_queue_server's destructor");
			exit(1);
		}
	}

};

//----------------------------------------------------------------------------

struct msg_queue_client
{
	char service_num;
	key_t key;//queue key
	int qid;//queue id
	string_msg_buf msg;//for holding the current msg

	msg_queue_client()
	{
		service_num = 'a';
		key = ftok(MSG_DIR, service_num);
		if((qid = msgget(key, 0)) == -1) {
			perror("Error in msg_queue_client's constructor");
			exit(1);
		}
	}

	msg_queue_client(char serviceNo)
	{
		service_num = serviceNo;
		key = ftok(MSG_DIR, service_num);
		if((qid = msgget(key, 0)) == -1) {
			perror("Error in msg_queue_client's constructor");
			exit(1);
		}
	}

	bool send_msg(long type, const char *text)
	{
		msg.mtype = type;
		strcpy(msg.mtext, text);
		if((msgsnd(qid, (msgbuf *)&msg,
			strlen(msg.mtext)+1, IPC_NOWAIT)) == -1)
		{
			perror("Error in msg_queue_client's send_message()");
			return false;
		}
		return true;
	}

};

//----------------------------------------------------------------------------

struct msg_queue_notifier
{
	char service_num;
	key_t key;//queue key
	int qid;//queue id
	string_msg_buf msg;//for holding the current msg

	msg_queue_notifier()
	{
		service_num = 'b';
		key = ftok(MSG_DIR, service_num);
		if((qid = msgget(key, IPC_CREAT|IPC_EXCL|0660)) == -1) {
			perror("Error in msg_queue_notifier's constructor");
			exit(1);
		}
	}

	msg_queue_notifier(char serviceNo)
	{
		service_num = serviceNo;
		key = ftok(MSG_DIR, service_num);
		if((qid = msgget(key, IPC_CREAT|IPC_EXCL|0660)) == -1) {
			perror("Error in msg_queue_notifier's constructor");
			exit(1);
		}
	}

	bool send_msg(long type, const char *text)
	{
		msg.mtype = type;
		strcpy(msg.mtext, text);
		if((msgsnd(qid, (msgbuf *)&msg,
			strlen(msg.mtext)+1, IPC_NOWAIT)) == -1)
		{
			perror("Error in msg_queue_notifier's send_message()");
			return false;
		}
		return true;
	}

	~msg_queue_notifier()
	{
		if(msgctl(qid, IPC_RMID, 0) == -1)
		{
			perror("Error in msg_queue_notifier's destructor");
			exit(1);
		}
	}

};

//----------------------------------------------------------------------------

struct msg_queue_receiver
{
	char service_num;
	key_t key;//queue key
	int qid;//queue id
	string_msg_buf msg;//for holding the current msg

	msg_queue_receiver()
	{
		service_num = 'b';
		key = ftok(MSG_DIR, service_num);
		if((qid = msgget(key, 0)) == -1) {
			perror("Error in msg_queue_receiver's constructor");
			exit(1);
		}
	}

	msg_queue_receiver(char serviceNo)
	{
		service_num = serviceNo;
		key = ftok(MSG_DIR, service_num);
		if((qid = msgget(key, 0)) == -1) {
			perror("Error in msg_queue_receiver's constructor");
			exit(1);
		}
	}

	//first recv_msg(), then get_msg() if received

	bool recv_msg(long type)
	{
		//true: msg received, call get_msg() to get the msg
		//false: no msg available
		msg.mtype = type;
		if((msgrcv(qid, (msgbuf *)&msg,
				MAX_SEND_SIZE, type, IPC_NOWAIT))==-1)
		{
			if (errno != ENOMSG)
			{
				perror("Error in msg_queue_receiver's recv_message()");
				exit(1);
			}
			return false;
		}
		return true;
	}

	char* get_msg()
	{
		return msg.mtext;
	}

};

