#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXFLDS 200     /* maximum possible number of fields */
#define MAXFLDSIZE 32   /* longest possible field + 1, define a 31 max field */
#define NUMFLDS 42		/* numver of fields for one record */
/* =========================================================================== 

toNumCSV.c 

output a CSV file with every field transformed to number

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
	/* constant headers 18 customer features and 24 products */
	const char* headers[NUMFLDS] = {"fetchDate", "cusId", "employeeIdx", "cntryOfResidence", "sex", 
							"age", "FirstContract", "newCusIdx", "seniority", "cusType", 
							"relationType", "foreignIdx", "chnlEnter", "deceasedIdx", "provCode",
							"activIdx", "income", "segment", 
							"savingAcnt", "guarantees", "currentAcnt", "derivativeAcnt", "payrollAcnt",
							"juniorAcnt", "moreParticularAcnt", "particularAcnt", "particularPlusAcnt", "shortDeposit",
							"mediumDeposit", "longDeposit", "eAcnt", "funds", "mortgage",
							"pensions", "loans", "taxes", "creditCard", "securities",
							"homeAcnt", "payroll", "payrollPensions", "directDebit"};

	/* Open the original file on cammand line and read it */
	char tmp[1024] = {0x0};
	int fldcnt = 0;
	char arr[MAXFLDS][MAXFLDSIZE] = {0x0};
	int recordcnt = 0;
	int i = 0;	
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
	printf("\n Creating %s.csv file\n",filename);
	FILE* fp;
 	
	fp = fopen(filename,"w+");

	/* print header */
	for(i = 0; i < NUMFLDS; i++){
		fprintf(fp, "%s,", headers[i]);
	}
	fprintf(fp, "\n");

	/* Read the original csv file */
	while(fgets(tmp, sizeof(tmp),in) != 0) /* read a record */
	{
	    int year, month, day;
	    double value = 0;
	    long epochDate;
	    fldcnt = 0;
	    recordcnt++;
	    if(recordcnt<2){
	    	continue;
	    }
		//printf("Record #: %d\n",recordcnt);
		//fprintf(fp,"\t");

		parse(tmp,",",arr,&fldcnt);    /* dissemble record into fields */

		/* remove the record without income data */
	    if((int) atof(arr[16])== -1){
			continue;
		}

		if(fldcnt == 42){ 

			char* cp;
			char* c;
			for(i = 0; i < fldcnt; i++)
			{                              
				//printf("\tField # %d == %s\n",i,arr[i]); /* print each field */
				
				//fprintf(fp,",%s ",arr[i]); /* string */
				cp = arr[i];
				cp += 2;
				if (*cp -'A' > 0){ // alphabets
					if(strlen(cp) == 1){
						//printf("Single char\n");
						//printf("\t Field %d == %d\n", i, (int) *arr[i]);
						fprintf(fp,"%d,",(int) *arr[i]);
					}else{
						//printf("A str\n");
						int str = 0;
						int j = 0;
						c = cp;
						for(j = 0; j < strlen(cp); j++){
							str = 100 * str + (int) *c;
							c++;
						}
						//printf("\t Field %d == %d\n", i, str);
						fprintf(fp,"%d,",str);
					}
				}	
				else{ // numbers
					year =  (int) atof(arr[i]);
					cp += 4;
					if(*cp - '-' == 0){
						//printf("A date\n");
						cp++;
						month = (int) atof(cp);
						cp+=3;
						day = (int) atof(cp);
						struct tm t;
    					time_t t_of_day;

					    t.tm_year = year - 1900;
					    t.tm_mon = month - 1;           // Month, 0 - jan
					    t.tm_mday = day;          // Day of the month
					    t.tm_hour = 0;
					    t.tm_min = 0;
					    t.tm_sec = 0;
					    t.tm_isdst = 0;        // DST? 1 = yes, 0 = no, -1 = unknown
					    t_of_day = mktime(&t);
    					//printf("seconds since the Epoch: %ld\n", (long) t_of_day);
    					epochDate = (long)t_of_day / 86400; 
    					//printf("days since the Epoch: %ld\n", epochDate);
    					//printf("\t Field %d == %ld\n", i,epochDate);
    					fprintf(fp, "%ld,", epochDate);
					}else{
						value = (double)atof(arr[i]);
						if(i == 16){
							//printf("\t Field %d == %.2f\n", i,value);
							fprintf(fp, "%.2f,", value); // double to 2 dec
						}else{
							//printf("\t Field %d == %d\n", i,(int) atof(arr[i]));
							fprintf(fp,"%d,",(int) atof(arr[i])); // int
						}
						
					}
						
				}
			}
		}
		fprintf(fp,"\n");
	}

	/* Close the original csv file */
	printf("Finish reading input csv file.\n");
    fclose(in);

    /* Close the new csv file */
    fclose(fp);
	printf("\n %s file created.\n",filename);

    return 0;	
}