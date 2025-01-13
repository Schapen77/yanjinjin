import json
import re
from openai import OpenAI


with open('/home/yanjinjin/lm-evaluation-harness/lm_eval/tasks/tool_call/function_call.json', 'r', encoding='utf-8') as f:
    function_call_definitions = json.load(f)


def get_tool_definition(tool_name, function_call_definitions):
    for tool in function_call_definitions:
        if tool['function']['name'] == tool_name:
            return tool
    return None


def load_dialogue_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def call_api(dialogue_history, prompt=None):
    client = OpenAI(
        api_key="67b0e1dd6f97cf9cb2061d7397411cc8.W9LO3INn9QDCD9iu",
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )

    if prompt:
        if dialogue_history[0]['role'] == 'system':
            dialogue_history[0]['content'] += "\n\n" + prompt
        else:
            dialogue_history.insert(0, {"role": "system", "content": prompt})

    try:
        messages = dialogue_history
        completion = client.chat.completions.create(
            model="glm-4",
            messages= messages
        )
        print(completion)
        return completion.choices[0].message.content

    except Exception as e:
        print(f"API调用失败，错误信息: {e}")
        return None


def evaluate_response(model_response, expected_tool_name):
    if model_response is None:
        return 0.0

    # 工具调用评估
    tool_call_score = 0.0
    function_pattern = re.search(r'✿FUNCTION✿:\s*([a-zA-Z0-9_]+)', model_response)
    tool_called = bool(function_pattern and function_pattern.group(1).strip())

    tool_definition = get_tool_definition(expected_tool_name, function_call_definitions)

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
        if expected_tool_name in model_response:
            print(f"工具调用正确，调用的工具是: {expected_tool_name}")
            tool_call_score = 1.0
        else:
            print(f"工具调用错误，期望调用: {expected_tool_name}，但模型没有调用该工具")
            tool_call_score = 0.0

    print(f"tool_called: {tool_called}")

    # 参数匹配评估
    param_score = 0.0
    total_params = 0
    if tool_called:
        params_match = re.search(r'["\']arguments["\']:\s*{(.+?)}|arguments:\s*{(.+?)}|✿ARGS✿:\s*{(.+?)}|Action Input:\s*{(.+?)}', model_response, re.DOTALL)
        
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

        if tool_definition:
            required_params = set(tool_definition['function']['parameters']['required'])
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

    print(f"工具调用分数：{tool_call_score}")
    print(f"工具调用分数：{param_score}")
    
    if total_params == 0:
        final_score = tool_call_score
    else:
        final_score = (tool_call_score + param_score) / 2

    return final_score


def run_evaluation(data_file, prompt):
    dialogues = load_dialogue_data(data_file)
    results = []
    total_score = 0.0

    for idx, item in enumerate(dialogues):
        print(f"正在评测第 {idx+1} 条数据...")
        
        dialogue_history = item['dialogue']
        model_response = call_api(dialogue_history, prompt=prompt)
        print(f"模型生成的结果: {model_response}")

        evaluation = evaluate_response(model_response, item.get("label", None))
        total_score += evaluation

        results.append({
            "dialogue_id": idx + 1,
            "dialogue_history": dialogue_history,
            "model_response": model_response,
            "evaluation": evaluation
        })

        print(f"对话ID {idx + 1} 的得分: {evaluation:f}")
        print("\n"*2)
    
    overall_accuracy = total_score / len(dialogues)
    print(f"\n准确率: {overall_accuracy:f}")
    
    results.append({"overall_accuracy": overall_accuracy})
    with open("/home/yanjinjin/lm-evaluation-harness/lm_eval/tasks/tool_call/evaluation_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    data_file = '/home/yanjinjin/Datasets/工具/不使用工具.json'
    prompt = """根据当前对话内容判断是否需要调用工具，如果需要，请按格式生成正确的工具调用。
    # 工具
    以下是你可以使用的工具列表：

    wechat_id_query: 如果用户表示没收到微信邀请，调用这个工具，再次发送加微邀请。
    - Parameters: {"type": "object", "properties": {"wechat_id": {"description": "微信号", "type": "string"}}, "required": ["wechat_id"]}
    - Format the arguments as a JSON object.

    confirm_follow_up_time: 如果用户表示自己现在不方便打电话，调用这个工具，确认下次回访时间。
    - Parameters: {"type": "object", "properties": {"Timing_of_follow_up": {"description": "下次回访时间", "type": "string"}}, "required": ["Timing_of_follow_up"]}
    - Format the arguments as a JSON object.

    non_local_store_query: 如果用户表示自己不在当地，请调用这个工具，查看地址周围是否有门店。
    - Parameters: {"type": "object", "properties": {"city": {"description": "城市名", "type": "string"}}, "required": ["city"]}
    - Format the arguments as a JSON object.

    contact_other_number: 如果客户表示需要联系别的号码，请调用这个工具，确认客户需要联系的号码是哪一个。
    - Parameters: {"type": "object", "properties": {"MobilePhone": {"description": "手机号码", "type": "string"}}, "required": ["MobilePhone"]}
    - Format the arguments as a JSON object.

    product_inventory: 如果客户想在指定门店购买特定产品，请调用这个工具，查看特定产品在指定门店是否有库存。
    - Parameters: {"type": "object", "properties": {"Store": {"description": "门店名", "type": "string"}, "Goods": {"description": "商品名", "type": "string"}}, "required": ["Store", "Goods"]}
    - Format the arguments as a JSON object.

    address_query: 如果用户询问门店地址，请调用这个工具，查看地址附近的门店地址并生成参数。
    - Parameters: {"type": "object", "properties": {"city": {"description": "城市名", "type": "string"}, "district": {"description": "区县名", "type": "string"}}, "required": ["city", "district"]}
    - Format the arguments as a JSON object.


    # 格式
    请根据当前对话内容判断是否需要调用工具。当需要调用工具时，请在回答中插入以下命令格式：
    ✿FUNCTION✿: 你要使用的工具名，仅限于 ['wechat_id_query', 'confirm_follow_up_time', 'non_local_store_query', 'contact_other_number', 'product_inventory', 'address_query'] 中的一个。
    ✿ARGS✿: 使用工具的参数，格式为 JSON。
    ✿RESULT✿: 工具返回的结果。
    ✿RETURN✿: 基于工具结果的回复。

    若不需要调用工具，请直接生成对话回答。"""
    run_evaluation(data_file, prompt)

