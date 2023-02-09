import json
import os
import threading
import uuid
from pathlib import Path
from urllib.parse import parse_qs
from datasets import load_dataset
import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import Repository
import random

from utils import force_git_push


# These variables are for storing the MTurk HITs in a Hugging Face dataset.
if Path(".env").is_file():
    load_dotenv(".env")
DATASET_REPO_URL = os.getenv("DATASET_REPO_URL")
FORCE_PUSH = os.getenv("FORCE_PUSH")
HF_TOKEN = os.getenv("HF_TOKEN")
PROMPT_TEMPLATES = Path("prompt_templates")

DATA_FILENAME = "data.jsonl"
DATA_FILE = os.path.join("data", DATA_FILENAME)
repo = Repository(local_dir="data", clone_from=DATASET_REPO_URL, use_auth_token=HF_TOKEN)
ds = load_dataset("HuggingFaceH4/instruction-pilot-outputs", split="train", use_auth_token=HF_TOKEN)

TOTAL_CNT = 10  # How many user inputs per HIT

# This function pushes the HIT data written in data.jsonl to our Hugging Face
# dataset every minute. Adjust the frequency to suit your needs.
PUSH_FREQUENCY = 60


def asynchronous_push(f_stop):
    if repo.is_repo_clean():
        print("Repo currently clean. Ignoring push_to_hub")
    else:
        repo.git_add(auto_lfs_track=True)
        repo.git_commit("Auto commit by space")
        if FORCE_PUSH == "yes":
            force_git_push(repo)
        else:
            repo.git_push()
    if not f_stop.is_set():
        # call again in 60 seconds
        threading.Timer(PUSH_FREQUENCY, asynchronous_push, [f_stop]).start()


f_stop = threading.Event()
asynchronous_push(f_stop)

demo = gr.Blocks()

def random_sample_with_least_annotated_examples_first():
    annotations = open(DATA_FILE, "r").readlines()
    id_to_count = {}
    for line in annotations:
        annotation = json.loads(line)
        # Only include annotations by actual turkers in the count.
        if annotation["assignmentId"] != "":
            example_id = annotation["id"]
            id_to_count[example_id] = id_to_count.get(example_id, 0) + 1
    ds_with_annotation_counts = ds.map(lambda example: {"annotation_count": id_to_count.get(example["id"], 0)})
    ds_with_annotation_counts = ds_with_annotation_counts.shuffle()
    ds_with_annotation_counts = ds_with_annotation_counts.sort("annotation_count")
    example = ds_with_annotation_counts.select([0])[0]
    # We only want to give the annotator 2 choices, so we sample 2 outputs without replacement.
    example["outputs"] = random.sample(example["outputs"], 2)
    return example

def prompt_pretty_markdown(prompt):
    prompt = prompt.replace("Input:", "\n\n<b>Input:</b>\n\n")
    return prompt


with demo:
    dummy = gr.Textbox(visible=False)  # dummy for passing assignmentId

    initial_sample = random_sample_with_least_annotated_examples_first()

    # We keep track of state as a JSON
    state_dict = {
        "taskId": str(uuid.uuid4()),
        "assignmentId": "",
        "cnt": 0,
        "data": [initial_sample],
    }
    state = gr.JSON(state_dict, visible=False)

    gr.Markdown("# Choose the more helpful response for the input")
    gr.Markdown("By 'helpful', we mean whatever answer you personally find more useful.")

    def _select_response(selected_response, state, dummy):
        if selected_response == "":
            # Don't do anything if the worker didn't select things yet.
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                state,
                dummy,
            )
        state["cnt"] += 1
        state_display = f"Messages left in HIT: {state['cnt']}/{TOTAL_CNT}"
        done = state["cnt"] == TOTAL_CNT
        state["data"][-1]["selected_response"] = selected_response
        if state["cnt"] == TOTAL_CNT:
            # Write the HIT data to our local dataset because the worker has
            # submitted everything now.
            with open(DATA_FILE, "a") as jsonlfile:
                json_data_with_assignment_id = [
                    json.dumps(
                        dict(
                            {"assignmentId": state["assignmentId"], "taskId": state["taskId"]},
                            **datum,
                        )
                    )
                    for datum in state["data"]
                ]
                jsonlfile.write("\n".join(json_data_with_assignment_id) + "\n")
        query = parse_qs(dummy[1:])
        if "assignmentId" in query and query["assignmentId"][0] != "ASSIGNMENT_ID_NOT_AVAILABLE":
            # It seems that someone is using this app on mturk. We need to
            # store the assignmentId in the state before submit_hit_button
            # is clicked. We can do this here in _predict. We need to save the
            # assignmentId so that the turker can get credit for their HIT.
            state["assignmentId"] = query["assignmentId"][0]
            toggle_final_submit = gr.update(visible=done)
            toggle_final_submit_preview = gr.update(visible=False)
        else:
            toggle_final_submit_preview = gr.update(visible=done)
            toggle_final_submit = gr.update(visible=False)

        toggle_submit_response_button = gr.update(visible=not done)

        new_sample = random_sample_with_least_annotated_examples_first()
        new_outputs = [obj["output"] for obj in new_sample["outputs"]]
        state["data"].append(new_sample)
        past_conversation = gr.update(
            value=prompt_pretty_markdown(new_sample["prompt"])
        )
        select_response = gr.update(choices=["(a) " + new_outputs[0], "(b) " + new_outputs[1], "(c) Both (a) and (b) are similarly good", "(d) Both (a) and (b) are similarly bad"], value="")

        return (
            past_conversation,
            select_response,
            toggle_submit_response_button,
            toggle_final_submit,
            toggle_final_submit_preview,
            state_display,
            state,
            dummy,
        )

    # Input fields
    gr.Markdown('<span style="padding:7px;color:black;background:#ffd21e;border-radius:10px"><b>Prompt</b></span>')

    past_conversation = gr.Markdown(
        value=prompt_pretty_markdown(initial_sample["prompt"])
    )
    initial_outputs = [obj["output"] for obj in initial_sample["outputs"]]

    gr.Markdown('<span style="padding:7px;color:black;background:#ffd21e;border-radius:10px"><b>Select the most helpful response</b></span>')
    select_response = gr.Radio(
        choices=["(a) " + initial_outputs[0], "(b) " + initial_outputs[1], "(c) Both (a) and (b) are similarly good", "(d) Both (a) and (b) are similarly bad"], label="",
    )

    submit_response_button = gr.Button("Submit Response")
    submit_hit_button = gr.Button("Submit HIT", visible=False)
    submit_hit_button_preview = gr.Button(
        "Submit Work (preview mode; no MTurk HIT credit, but your examples will still be stored)",
        visible=False,
    )

    state_display = gr.Markdown(f"Messages left in HIT: 0/{TOTAL_CNT}")

    # Button event handlers
    get_window_location_search_js = """
        function(select_response, state, dummy) {
            return [select_response, state, window.location.search];
        }
        """

    submit_response_button.click(
        _select_response,
        inputs=[select_response, state, dummy],
        outputs=[
            past_conversation,
            select_response,
            submit_response_button,
            submit_hit_button,
            submit_hit_button_preview,
            state_display,
            state,
            dummy,
        ],
        _js=get_window_location_search_js,
    )

    post_hit_js = """
        function(state) {
            // If there is an assignmentId, then the submitter is on mturk
            // and has accepted the HIT. So, we need to submit their HIT.
            const form = document.createElement('form');
            form.action = 'https://workersandbox.mturk.com/mturk/externalSubmit';
            form.method = 'post';
            for (const key in state) {
                const hiddenField = document.createElement('input');
                hiddenField.type = 'hidden';
                hiddenField.name = key;
                hiddenField.value = state[key];
                form.appendChild(hiddenField);
            };
            document.body.appendChild(form);
            form.submit();
            return state;
        }
        """

    submit_hit_button.click(
        lambda state: state,
        inputs=[state],
        outputs=[state],
        _js=post_hit_js,
    )

    refresh_app_js = """
        function(state) {
            // The following line here loads the app again so the user can
            // enter in another preview-mode "HIT".
            window.location.href = window.location.href;
            return state;
        }
        """

    submit_hit_button_preview.click(
        lambda state: state,
        inputs=[state],
        outputs=[state],
        _js=refresh_app_js,
    )

demo.launch()
