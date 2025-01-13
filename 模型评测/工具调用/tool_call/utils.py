import re
import json

def process_tool_call_results(doc, results):

    tool_call_score = 0.0  # 工具调用的得分
    param_score = 0.0  # 参数生成的得分
    total_params = 0  # 总参数数
    tool_definition = None

    model_output = results[0]
    print(f"模型生成的结果: {model_output}")

    # 工具调用判断是否准确
    expected_tool_name = doc.get("label", None)
    print(f"期望调用的工具: {expected_tool_name}")

    function_pattern = re.search(r'✿FUNCTION✿:\s*([a-zA-Z0-9_]+)', model_output)
    print(f"function_pattern: {function_pattern}")
    tool_called = bool(function_pattern and function_pattern.group(1).strip())

    if expected_tool_name is None:
        print("不希望调用任何工具")

        if tool_called:
            print("模型调用了工具，但不应该调用工具")
            tool_call_score = 0.0
        else:
            print("模型没有调用工具，符合预期")
            tool_call_score = 1.0

    else:
        print(f"期望调用的工具: {expected_tool_name}")

        if expected_tool_name in model_output:
            print(f"工具调用正确，调用的工具是: {expected_tool_name}")
            tool_call_score = 1.0
        else:
            print(f"工具调用错误，期望调用: {expected_tool_name}，但模型没有调用该工具")
            tool_call_score = 0.0

    print(f"tool_called: {tool_called}")

    # 参数生成是否准确
    if tool_call_score == 1.0 and tool_called:

        params_match = re.search(r'["\']arguments["\']:\s*{(.+?)}|arguments:\s*{(.+?)}|✿ARGS✿:\s*{(.+?)}|Action Input:\s*{(.+?)}', model_output, re.DOTALL)

        if params_match:
            if params_match.group(1):
                params_str = params_match.group(1)
            elif params_match.group(2):
                params_str = params_match.group(2)
            elif params_match.group(3):
                params_str = params_match.group(3)
            elif params_match.group(4):
                params_str = params_match.group(4)           
        else:
            params_str = None
        print(f"params_str: {params_str}")
        
        model_params = {}
        if params_str:
            model_params = dict(re.findall(r'["\']([\w-]+)["\']:\s*["\'](.+?)["\']', params_str))


        with open('/home/yanjinjin/lm-evaluation-harness/lm_eval/tasks/tool_call/function_call.json', 'r', encoding='utf-8') as f:
            function_call = json.load(f)

        tool_definition = next((tool for tool in function_call if tool['function']['name'] == expected_tool_name), None)

        # 判断生成的参数是否匹配工具定义
        if tool_definition:
            required_params = set(tool_definition['function']['parameters']['properties'].keys()) if tool_definition else set()
            model_param_keys = set(model_params.keys())

            print(f"工具定义中需要的参数: {required_params}")
            print(f"模型生成的参数: {model_param_keys}")

            total_params = len(model_params)
            correct_params = 0

            for param_name in model_param_keys:
                if param_name in required_params:
                    correct_params += 1
        
            if total_params > 0:
                param_score = correct_params / total_params
        
            if param_score == 1.0:
                print("参数完全匹配")
            else:
                missing_params = required_params - model_param_keys
                extra_params = model_param_keys - required_params
                print(f"参数部分匹配，缺少参数: {missing_params}，多余的参数: {extra_params}")
        else:
            print("没有找到工具定义，跳过参数匹配")

################################################################################################

    # 计算最终得分
    if total_params == 0:
        final_score = tool_call_score
    else:
        final_score = (tool_call_score + param_score) / 2

    print(f"最终得分: {final_score}")
    return {"acc": float(final_score)}


################################################################################################