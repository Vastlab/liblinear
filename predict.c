#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "linear.h"

int print_null(const char *s,...) {return 0;}

static int (*info)(const char *fmt,...) = &printf;

struct feature_node *x;
int max_nr_attr = 64;

struct model* model_;
int flag_predict_probability=0;

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,(size_t)max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void do_predict(FILE *input, FILE *output)
{
	int correct = 0;
	int falsepos=0, falseneg=0, truepos=0, trueneg=0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int nr_class=get_nr_class(model_);
	double *prob_estimates=NULL;
	int j, n;
	int nr_feature=get_nr_feature(model_);
	double *dec_values = (double*)  malloc((size_t)(sizeof(double)* (size_t)model_->nr_class));
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;

	if(flag_predict_probability)
	{
		int *labels;

		if(!check_probability_model(model_))
		{
			fprintf(stderr, "probability output is only supported for logistic regression\n");
			exit(1);
		}

		labels=(int *) malloc((size_t)nr_class*sizeof(int));
		get_labels(model_,labels);
		prob_estimates = (double *) malloc((size_t)nr_class*sizeof(double));
		fprintf(output,"labels");
		for(j=0;j<nr_class;j++)
			fprintf(output," %d",labels[j]);
		fprintf(output,"\n");
		free(labels);
	}

	max_line_len = 1024;
	line = (char *)malloc((size_t)max_line_len*sizeof(char));
	while(readline(input) != NULL)
	{
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = 0; // strtol gives 0 if wrong format

		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(total+1);

		target_label = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);

		while(1)
		{
			if(i>=max_nr_attr-2)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct feature_node *) realloc(x,(size_t)max_nr_attr*sizeof(struct feature_node));
			}

			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);

			// feature indices larger than those in training are not used
			if(x[i].index <= nr_feature)
				++i;
		}

		if(model_->bias>=0)
		{
			x[i].index = n;
			x[i].value = model_->bias;
			i++;
		}
		x[i].index = -1;

		if(flag_predict_probability)
		{
			int j;
			predict_label = predict_probability(model_,x,prob_estimates);
			fprintf(output,"%g",predict_label);
			for(j=0;j<model_->nr_class;j++)
				fprintf(output," %g",prob_estimates[j]);
			fprintf(output,"\n");
		}
		else
		{
                  //			predict_label = predict(model_,x);
                  predict_label = predict_values(model_,x,dec_values);
                  int jj=0;
                  for(jj=0;jj< model_->nr_class; jj++){
                    if(model_->label[jj]==predict_label) break;
                  }
                  fprintf(output,"%g (%g): ",predict_label, target_label);
                  for(jj=0;jj< model_->nr_class; jj++){
                    fprintf(output,"%g ", dec_values[jj]);
                  }
                  fprintf(output," \n ");
		}
                int unknown=true;
                for(int jj=0;jj< model_->nr_class; jj++){
                  if(model_->label[jj] == target_label) {
                    unknown = false;
                    break;
                  }
                }

		if((predict_label == target_label) || (unknown &&  predict_label == -999999)){
                  ++correct;++truepos;}
                else ++falsepos;
                if(predict_label == -999999){
                  if(unknown) ++trueneg;
                  else ++falseneg;
                }

                error += (predict_label-target_label)*(predict_label-target_label);
                sump += predict_label;
                sumt += target_label;
                sumpp += predict_label*predict_label;
                sumtt += target_label*target_label;
                sumpt += predict_label*target_label;
                ++total;

	}
	if(check_regression_model(model_))
	{
		info("Mean squared error = %g (regression)\n",error/total);
		info("Squared correlation coefficient = %g (regression)\n",
			((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
			((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
			);
	}
	else
		info("Accuracy = %g%% (%d/%d)\n",(double) correct/total*100,correct,total);

        if ( (truepos+falsepos) > 0){
          double precision = ((double) (truepos)/(truepos+falsepos));
          double recall = 0;
          if((truepos + falseneg) > 0) recall = ((double) truepos)/(truepos + falseneg);
          double fmeasure = 0;
          if( (precision + recall > 0)) fmeasure = 2* precision*recall/(precision + recall);
          printf("  Precision=%lf,   Recall=%lf   Fmeasure=%lf\n",precision, recall, fmeasure);
          printf("   Total tests=%d, True pos %d True Neg %d, False Pos %d, False neg %d\n",
                 truepos+ trueneg+ falsepos+ falseneg, truepos, trueneg, falsepos, falseneg);
        } else if(((truepos+falsepos)==0)){
          printf("  Precision=0,   Recall=0   Fmeasure=0\n");
        }
	if(flag_predict_probability)
		free(prob_estimates);
}

void exit_with_help()
{
	printf(
	"Usage: predict [options] test_file model_file output_file\n"
	"options:\n"
	"-b probability_estimates: whether to output probability estimates, 0 or 1 (default 0); currently for logistic regression only\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

int main(int argc, char **argv)
{
	FILE *input, *output;
	int i;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
			case 'b':
				flag_predict_probability = atoi(argv[i]);
				break;
			case 'q':
				info = &print_null;
				i--;
				break;
			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}
	if(i>=argc)
		exit_with_help();

	input = fopen(argv[i],"r");
	if(input == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",argv[i]);
		exit(1);
	}

	output = fopen(argv[i+2],"w");
	if(output == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",argv[i+2]);
		exit(1);
	}

	if((model_=load_model(argv[i+1]))==0)
	{
		fprintf(stderr,"can't open model file %s\n",argv[i+1]);
		exit(1);
	}

	x = (struct feature_node *) malloc((size_t)max_nr_attr*sizeof(struct feature_node));
	do_predict(input, output);
	free_and_destroy_model(&model_);
	free(line);
	free(x);
	fclose(input);
	fclose(output);
	return 0;
}

