#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXFLDS 200     /* maximum possible number of fields */
#define MAXFLDSIZE 32   /* longest possible field + 1, define a 31 max field */

/* =======================================================================+======= 

readCSV.c 

read and parse the CSV file and output the records and fields to the command line.

================================================================================= */


void parse(char *record, char *delim, char arr[][MAXFLDSIZE],int *fldcnt)
{
    char* p = strtok(record, delim);
    int fld = 0;
    
    while(p)
    {
        strcpy(arr[fld], p);
		fld++;
		p = strtok(NULL, delim);
	}		
	*fldcnt = fld;
}


int main(int argc, char *argv[])
{
	
	/* Open the original file on cammand line and read it */
	char tmp[1024] = {0x0};
	int fldcnt = 0;
	char arr[MAXFLDS][MAXFLDSIZE] = {0x0};
	int recordcnt = 0;	
	FILE *in = fopen(argv[1],"r");         /* get input file as command line arg */
	
	if(in == NULL)
	{
		perror("File open error");
		exit(EXIT_FAILURE);
	}


	/* Read the original csv file */
	//int head = 0;
	while(fgets(tmp, sizeof(tmp),in) != 0) /* read a record */
	{
	    int i = 0;
	    recordcnt++;
		//printf("Record #: %d\n",recordcnt);

		parse(tmp,",",arr,&fldcnt);    /* dissemble record into fields */
		for(i = 0; i < fldcnt; i++){                              
			//if(head < 2)
				printf("\tField # %d == %s\n",i,arr[i]); /* print each field */
		}

		//head++;
	}

	/* Close the original csv file */
	printf("Finish reading input csv file.\n");
    fclose(in);


    return 0;	
}