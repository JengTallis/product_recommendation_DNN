#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXFLDS 200     /* maximum possible number of fields */
#define MAXFLDSIZE 32   /* longest possible field + 1, define a 31 max field */

/* =========================================================================== 

parseCSV.c 

parse the CSV file and output a new CSV file with cleaned-up records.

The following fields are removed:

9. indrel 
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
	    int a = 0;
	    int b = 0;
	    int flag = 0;
	    fldcnt = 0;
	    recordcnt++;
		//printf("Record #: %d\n",recordcnt);
		//fprintf(fp,"\n");
		fprintf(fp,"\t");

		parse(tmp,",",arr,&fldcnt);    /* dissemble record into fields */
		if(fldcnt >= 45 ){ /* a complete record has >=45 (48) fields */
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

				if((i+a+b != 9) && (i+a+b != 10) && (i+a+b != 13) && (i+a+b != 17) && (i+a+b != 19)){
					//if (flag>0)
						printf("\tField # %d == %s\n",i,arr[i]);
				
					fprintf(fp,",%s ",arr[i]); /* string */
				}

				else{
					//if (flag>0)
						printf("\t kill Field # %d == %s\n",i,arr[i]);
				}

				/*
				if (*arr[i]-'A' > 0)
					printf("\t Field %d == %d\n", i, (int) *arr[i]);
				else
					printf("\t Field %d == %d\n", i,(int) atof(arr[i]));
				
				printf("\tField # %d == %f\n",i,(double) atof(arr[i]));

				fprintf(fp, ",%f", (double) atof(arr[i])); // double

				*/
				
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