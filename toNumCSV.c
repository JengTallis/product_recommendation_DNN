#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXFLDS 200     /* maximum possible number of fields */
#define MAXFLDSIZE 32   /* longest possible field + 1, define a 31 max field */

/* =========================================================================== 

toNumCSV.c 

transform every field in the CSV file to number.

============================================================================== */

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

	/* Create a new csv file */
	char new_name[100];
	char* filename;
    printf("\n Enter the filename :");
	gets(new_name);

	filename = strcat(new_name,".csv");
	printf("\n Creating %s.csv file",filename);
	FILE* fp;
 	
	fp = fopen(filename,"w+");

	/* Read the original csv file */
	while(fgets(tmp, sizeof(tmp),in) != 0) /* read a record */
	{
	    int i = 0;
	    int year, month, day;
	    long epochDate;
	    fldcnt = 0;
	    recordcnt++;
		printf("Record #: %d\n",recordcnt);
		//fprintf(fp,"\n");
		fprintf(fp,"\t");

		parse(tmp,",",arr,&fldcnt);    /* dissemble record into fields */
		if(fldcnt == 42 ){ 
			char* cp;
			char* c;
			for(i = 0; i < fldcnt; i++)
			{                              
				printf("\tField # %d == %s\n",i,arr[i]); /* print each field */
				
				//fprintf(fp,",%s ",arr[i]); /* string */
				cp = arr[i];
				cp += 2;
				if (*cp -'A' > 0){ // alphabets
					if(strlen(cp) == 1){
						printf("Single char\n");
						printf("\t Field %d == %d\n", i, (int) *arr[i]);
					}else{
						printf("A str\n");
						int str = 0;
						int j = 0;
						c = cp;
						for(j = 0; j < strlen(cp); j++){
							str = 100 * str + (int) *c;
							c++;
						}
						printf("\t Field %d == %d\n", i, str);
					}
				}	
				else{ // numbers
					year =  (int) atof(arr[i]);
					cp += 4;
					if(*cp - '-' == 0){
						printf("A date\n");
						cp++;
						month = (int) atof(cp);
						cp+=3;
						day = (int) atof(cp);
						struct tm t;
    					time_t t_of_day;

					    t.tm_year = year-1900;
					    t.tm_mon = month;           // Month, 0 - jan
					    t.tm_mday = day;          // Day of the month
					    t.tm_hour = 0;
					    t.tm_min = 0;
					    t.tm_sec = 0;
					    t.tm_isdst = 0;        // DST? 1 = yes, 0 = no, -1 = unknown
					    t_of_day = mktime(&t);
    					printf("seconds since the Epoch: %ld\n", (long) t_of_day);
    					epochDate = (long)t_of_day/86400; 
    					printf("days since the Epoch: %ld\n", epochDate);
    					printf("\t Field %d == %ld\n", i,epochDate);
					}else{
						printf("\t Field %d == %d\n", i,(int) atof(arr[i]));
					}
					
					fprintf(fp, ",%f", (double)atof(arr[i])); // double	
				}
			}
		}
	}

	/* Close the original csv file */
	printf("Finish reading input csv file.\n");
    fclose(in);

    /* Close the new csv file */
    fclose(fp);
	printf("\n %s file created.\n",filename);

    return 0;	
}