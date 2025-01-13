import re

def get_next_task_id(expected_tasks, next_expected_task):
    if next_expected_task is None:
        return None
    
    for task in expected_tasks:
        if task.get("id") == next_expected_task:
            return task.get("id")

    return None

def is_force_end_task(expected_tasks, next_expected_task):
    for task in expected_tasks:
        if task["id"] == next_expected_task:
            return "结束对话" in task.get("description", "")
    return False

def determine_expected_status(task, force_end, no_task_needed):
    if no_task_needed:
        return "否"
    if force_end and task["description"] == "结束对话":
        return "是"
    return "是" if task["needs_action"] else "否"

def check_task_status_fill(expected_tasks, model_output, next_task_id, force_end, no_task_needed):
    status_correct = True 

    for task in expected_tasks:
        task_id = task.get("id").replace("任务", "")
        expected_status = determine_expected_status(task, force_end, no_task_needed)

        # 这里因为输出格式不同，所以匹配的规则不同，可以合并到一起。
        # 指令7
        pattern = f'[\"“]?(是否还需要进行任务\\s*{task_id}\\s*（[^）]*）)[\"”]?[：:]\\s*[\"“]?(是|否)[\"”]?'

        # 指令4
        # question_text = f"是否还需要进行任务{task_id}（[^）]*）"
        # pattern = rf'{{\s*["\']answer["\']\s*[:：]\s*["\'](是|否)["\'],\s*["\']question["\']\s*[:：]\s*["\']{question_text}["\']\s*}}'
        
        match = re.search(pattern, model_output)

        if match:
            # 这里也是因为输出不同，所以提取的部分有差异。
            #指令4
            # model_status = match.group(1)
            #指令7
            model_status = match.group(2)
            if model_status != expected_status:
                print(f"任务{task_id}状态填写错误：期望 '{expected_status}'，模型填写 '{model_status}'")
                status_correct = False
        else:
            print(f"任务{task_id}在模型输出中未找到对应的 question-answer 对")
            status_correct = False

    return 1.0 if status_correct else 0.0

def process_results(doc, results):

    model_output = results[0]
    print(f"\n模型生成的结果: {model_output}")

    task_planning = 0.0
    task_status_fill = 0.0

    # 从模型输出中提取任务规划和任务状态
    match_task_id = re.search(r'["“]?将要进行的下一个任务\s*id["”]?\s*[:：]\s*["“]?(\w+)["”]?', model_output)
    print(f"\n模型任务规划: {match_task_id}")
    match_task_complete = re.search(r'是否完成所有任务：(.+)', model_output)
    print(f"\n模型任务状态: {match_task_complete}")

    # 获取期望
    expected_tasks = doc.get("label", {}).get("tasks", [])
    next_expected_task = doc.get("label", {}).get("next_expected_task")
    next_expected_task_id = get_next_task_id(expected_tasks, next_expected_task)

    # 确定是否需要提前结束
    force_end = is_force_end_task(expected_tasks, next_expected_task)
    no_task_needed = next_expected_task_id is None and not force_end

    # 判断下一个任务id是否准确
    if match_task_id:
        number = match_task_id.group(1).strip()
        if number and number.lower() != "none":
            predicted_task_id = "任务" + number
        else:
            predicted_task_id = "none"
        
        next_expected_task_id = next_expected_task_id if next_expected_task_id is not None else "none"

        print(f"\n模型预测的下一个任务ID: {predicted_task_id}")
        print(f"\n期望的下一个任务ID: {next_expected_task_id}")

        if predicted_task_id == next_expected_task_id:
            task_planning = 1.0
            print("\n任务规划正确：模型成功识别下一个需要进行的任务")
        else:
            task_planning = 0.0
            print("\n任务规划错误：模型未正确识别下一个需要进行的任务")
    
    # 判断任务状态填写是否正确
    if match_task_id and match_task_id.group(1).lower() == 'none' and task_planning == 1.0:
        task_status_fill = 1.0
    else:
        task_status_fill = check_task_status_fill(expected_tasks, model_output, next_expected_task_id, force_end, no_task_needed)

    print(f"\n任务规划得分: {task_planning}")
    print(f"\n任务状态填写得分: {task_status_fill}")
    
    # 计算最终得分
    final_score = (task_planning + task_status_fill) / 2
    print(f"\n最终得分: {final_score}")
    return {"acc": float(final_score)}