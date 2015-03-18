#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#ifdef __cplusplus
extern "C" {
#endif

struct feature_node
{
	int index;
	double value;
};

struct problem
{
	int l, n;
	double *y;
	struct feature_node **x;
	double bias;            /* < 0 if no bias term */  
};

enum { L2R_LR, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, L1R_L2LOSS_SVC, L1R_LR, L2R_LR_DUAL, L2R_L2LOSS_SVR = 11, L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL }; /* solver_type */

typedef enum {OPT_PRECISION, OPT_RECALL,  OPT_FMEASURE,  OPT_HINGE, OPT_BALANCEDRISK}  openset_optimization_t;

struct parameter
{
	int solver_type;

	/* these are for training only */
	double eps;	        /* stopping criteria */
	double C;
	int nr_weight;
	int *weight_label;
	double* weight;
	double p;

	int do_open;	/* do we want to do open-set expansion of base kernel */
        openset_optimization_t optimize; /* choice of what to optimize */ 
	bool exaustive_open; /* do we do exaustive optimization for openset.. default is false */
        double beta; /* for use in f-measure optimization */ 
        double near_preasure, far_preasure; /* for openset risk preasures */
        FILE* vfile; /* for logging verbose stuff during debugging */ 
        int  rejectedID; /* id for rejected classes (-99999 is the default) */ 
};

struct model
{
	struct parameter param;
	int nr_class;		/* number of classes */
	int nr_feature;
	double *w;
	int *label;		/* label of each class */
	double bias;

        int openset_dim;        /* dimension of data for 1-vs-set models,  if 1-vs-set then openset_dim=nr_class-1*/
        double *alpha, *omega;  /* planes offsets for 1-vs-set   alpha[openset_dim], omega[openset_dim] */

};

struct model* train(const struct problem *prob, const struct parameter *param);
void cross_validation(const struct problem *prob, const struct parameter *param, int nr_fold, double *target);

double predict_values(const struct model *model_, const struct feature_node *x, double* dec_values);
double predict(const struct model *model_, const struct feature_node *x);
double predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates);

int save_model(const char *model_file_name, const struct model *model_);
struct model *load_model(const char *model_file_name);


int get_nr_feature(const struct model *model_);
int get_nr_class(const struct model *model_);
void get_labels(const struct model *model_, int* label);
double get_decfun_coef(const struct model *model_, int feat_idx, int label_idx);
double get_decfun_bias(const struct model *model_, int label_idx);

void free_model_content(struct model *model_ptr);
void free_and_destroy_model(struct model **model_ptr_ptr);
void destroy_param(struct parameter *param);

const char *check_parameter(const struct problem *prob, const struct parameter *param);
int check_probability_model(const struct model *model);
int check_regression_model(const struct model *model);
void set_print_string_function(void (*print_func) (const char*));

  void openset_analyze_set(const struct problem &prob,  struct model *model_,  const struct parameter *param);

  struct model *convertto_onevset_model(struct model *basemodel, const struct problem *prob, const struct parameter *param); /* takes a model in,  generates onevset model from it) */



#ifdef __cplusplus
}
#endif

#endif /* _LIBLINEAR_H */

