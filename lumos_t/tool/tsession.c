#include "tsession.h"

void run_benchmarks(char *benchmark, int coretype)
{
    cJSON *cjson_benchmark = NULL;
    cJSON *cjson_public = NULL;
    cJSON *cjson_interface = NULL;
    cJSON *cjson_benchmarks = NULL;
    cJSON *cjson_params = NULL;
    cJSON *cjson_compares = NULL;
    cJSON *cjson_benchmark_item = NULL;
    cJSON *cjson_param_item = NULL;
    cJSON *cjson_compare_item = NULL;
    cJSON *cjson_single_benchmark = NULL;
    cJSON *cjson_benchmark_value = NULL;
    char *interface = NULL;
    char *type = NULL;
    char **benchmarks = NULL;
    char **params = NULL;
    char **compares = NULL;
    int benchmark_num = 0;
    int param_num = 0;
    int compare_num = 0;
    void **space = NULL;
    void **ret = NULL;
    int all_pass = 0;
    char *json_benchmark = load_from_json_file(benchmark);
    cjson_benchmark = cJSON_Parse(json_benchmark);
    cjson_public = cJSON_GetObjectItem(cjson_benchmark, "Public");
    cjson_interface = cJSON_GetObjectItem(cjson_public, "interface");
    cjson_benchmarks = cJSON_GetObjectItem(cjson_public, "benchmarks");
    cjson_params = cJSON_GetObjectItem(cjson_public, "params");
    cjson_compares = cJSON_GetObjectItem(cjson_public, "compares");
    interface = cjson_interface->valuestring;
    benchmark_num = cJSON_GetArraySize(cjson_benchmarks);
    param_num = cJSON_GetArraySize(cjson_params);
    compare_num = cJSON_GetArraySize(cjson_compares);
    benchmarks = malloc(benchmark_num*sizeof(char*));
    params = malloc(param_num*sizeof(char*));
    compares = malloc(compare_num*sizeof(char*));
    space = malloc((param_num+1)*sizeof(void*));
    ret = malloc(compare_num*sizeof(void*));
    for (int i = 0; i < benchmark_num; ++i){
        cjson_benchmark_item = cJSON_GetArrayItem(cjson_benchmarks, i);
        benchmarks[i] = cjson_benchmark_item->valuestring;
    }
    for (int i = 0; i < param_num; ++i){
        cjson_param_item = cJSON_GetArrayItem(cjson_params, i);
        params[i] = cjson_param_item->valuestring;
    }
    for (int i = 0; i < compare_num; ++i){
        cjson_compare_item = cJSON_GetArrayItem(cjson_compares, i);
        compares[i] = cjson_compare_item->valuestring;
    }
    test_run(interface, coretype);
    for (int i = 0; i < benchmark_num; ++i){
        int compare_flag = 1;
        cjson_single_benchmark = cJSON_GetObjectItem(cjson_benchmark, benchmarks[i]);
        load_params(cjson_single_benchmark, params, space, param_num);
        cjson_benchmark_value = cJSON_GetObjectItem(cjson_single_benchmark, "benchmark");
        // 运行测试
        int run_flag = 1;
        if (coretype == GPU){
            run_flag = call_cu(interface, space, ret);
        } else {
            run_flag = call(interface, space, ret);
        }
        if (run_flag){
            compare_flag = compare_test(cjson_benchmark_value, ret, compares, compare_num);
            if (compare_flag == 0){
                test_msg_pass(benchmarks[i]);
            }
            else{
                all_pass = 1;
                test_msg_error(benchmarks[i]);
            }
        }
        else {
            all_pass = 1;
            test_msg_error("Interface can't find, Please checkout your testlist");
            continue;
        }
    }
    test_res(all_pass, " ");
}

void run_all_benchmarks(char *benchmarks, int coretype)
{
    FILE *fp = fopen(benchmarks, "r");
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *tmp = (char*)malloc(file_size * sizeof(char));
    memset(tmp, '\0', file_size * sizeof(char));
    fseek(fp, 0, SEEK_SET);
    fread(tmp, sizeof(char), file_size, fp);
    fclose(fp);
    int head = 0;
    for (int i = 0; i < file_size; ++i){
        if (tmp[i] == '\n'){
            tmp[i] = '\0';
            run_benchmarks(tmp+head, coretype);
            head = i+1;
        }
    }
}

void release_params_space(void **space, int num)
{
    for (int i = 0; i < num; ++i){
        free(space[i]);
    }
}
