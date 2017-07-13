#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXFLDS 200     /* maximum possible number of fields */
#define MAXFLDSIZE 32   /* longest possible field + 1, define a 31 max field */

#define NUMFLDS 42

/* =========================================================================== 

parseCSV.c 

parse the CSV file and output a new CSV file with cleaned-up records.

The following fields are removed:

09. indrel 
10. ult_fec_cl_1t
13. indresi
15. conyuemp
18. tipodom
20. nomprov

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
	// char header[NUMFLDS][MAXFLDSIZE]={0x0};
	// int c = 0;
	// for(c = 0; c < NUMFLDS; c++){
	// 	strcpy(header[c], headers[c]);
	// }

	/* Open the original file on cammand line and read it */
	char tmp[1024] = {0x0};
	int fldcnt = 0;
	int i, head = 0;
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
	printf("\n Creating %s file \n",filename);
	FILE* fp;
 	
	fp = fopen(filename,"w+");

	/* print header row */
	if(fgets(tmp, sizeof(tmp),in) != 0){
		fprintf(fp, ",");
		for(i = 0; i < NUMFLDS; i++){
			fprintf(fp, " %s, ", headers[i]);
		}
		head++;
		recordcnt++;
		fprintf(fp, "\n");
		fprintf(fp, ",");
	}

	/* Read the original csv file */
	while(fgets(tmp, sizeof(tmp),in) != 0) /* read a record */
	{
	    i = 0;
	    int a = 0;
	    int b = 0;
	    int flag = 0;
	    fldcnt = 0;
	    recordcnt++;
		//printf("Record #: %d\n",recordcnt);
		//fprintf(fp,"\n");
		//fprintf(fp, "\t");

		parse(tmp, ",", arr, &fldcnt);    /* dissemble record into fields */
		if (head != 0) // data rows
		{
			if(fldcnt >= 45){ /* a complete record has >=45 (48) fields */ // >=45
				for(i = 0; i < fldcnt; i++)
				{                              
					//printf("\tField # %d == %s\n",i,arr[i]); /* print each field */
					//fprintf(fp,",%s ",arr[i]); /* string */
					if(i == 9){
						if ((int)atoi(arr[i]) == 1){
							a = 1;
						}
					}
					if(i+a == 20){
						char* c = arr[i];
						c++;
						//printf("%c\n", *c);

						if(*c - 'A'< 0){
							//printf("number %c\n", *c);
							b = 1;
							flag++;
						}else{
							//printf("alphabet %c\n", *c);
							flag=0;
						}
					}

					if (i+a+b == 22){
						char* c = arr[i];
						if(*c - '0'== 0){
							//printf("\tField # %d == %s\n",i,arr[i]);
							fprintf(fp, " %s, ", "-1");
						}
					}

					if((i+a+b != 9) && (i+a+b != 10) && (i+a+b != 13) && (i+a+b != 17) && (i+a+b != 19) && (i+a+b != 20)){
						//printf("\tField # %d == %s\n",i,arr[i]);
						fprintf(fp," %s, ",arr[i]); /* string */
					}else{
						//if (flag>0)
							//printf("\t kill Field # %d == %s\n",i,arr[i]);
					}	
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