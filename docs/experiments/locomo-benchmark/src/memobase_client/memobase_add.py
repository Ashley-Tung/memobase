import uuid
import json
import time
import os
import sys
import threading
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from memobase import MemoBaseClient, AsyncMemoBaseClient, ChatBlob

load_dotenv()

root_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(root_dir, "config.yaml")


def string_to_uuid(s: str, salt="memobase_client") -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s + salt))


class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class MemobaseADD:
    def __init__(self, data_path=None, batch_size=2, reprocess=False):
        self.client = MemoBaseClient(
            api_key=os.getenv("MEMOBASE_API_KEY"),
            project_url=os.getenv("MEMOBASE_PROJECT_URL", "https://api.memobase.dev"),
        )
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = f.read()
            print(config)
            self.client.update_config(config)
            print("Config updated successfully")
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.reprocess = reprocess
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    def add_memory(self, user, message, retries=3):
        for attempt in range(retries):
            try:
                _ = user.insert(ChatBlob(messages=message))
                return
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)  # Wait before retrying
                    continue
                else:
                    raise e

    def add_memories_for_speaker(self, speaker, messages, desc):
        real_uid = string_to_uuid(speaker)
        u = self.client.get_or_create_user(real_uid)
        for i in tqdm(range(0, len(messages), self.batch_size), desc=desc):
            # IMPORTANT: default batch size is 2 so this is
            # Two lines of ocnversations, likely from two different users
            # the two messages are added as a single ChatBlob object
            batch_messages = messages[i : i + self.batch_size]
            self.add_memory(u, batch_messages)
        u.flush()
        
        # Save profile after all memories are processed
        os.makedirs("memobase_memories", exist_ok=True)
        try:
            profile_data = u.profile(need_json=True)
            with open(f"memobase_memories/{u.user_id}.json", "w") as f:
                json.dump(profile_data, f, indent=2, cls=UUIDEncoder)
            print(f"Profile saved for user {u.user_id}")
        except Exception as e:
            print(f"Failed to save profile for user {u.user_id}: {e}")

    def process_conversation(self, item, idx):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        # delete all memories for the two users
        try:
            self.client.delete_user(string_to_uuid(speaker_a_user_id))
            self.client.delete_user(string_to_uuid(speaker_b_user_id))
        except Exception as e:
            pass

        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue

            date_time_key = key + "_date_time"
            timestamp = conversation[date_time_key]
            chats = conversation[key]

            messages = []
            messages_reverse = []
            for chat in chats:
                if chat["speaker"] == speaker_a:
                    messages.append(
                        {
                            "role": "user",
                            "content": chat["text"],
                            "alias": speaker_a,
                            "created_at": timestamp,
                        }
                    )
                    # messages_reverse.append(
                    #     {
                    #         "role": "assistant",
                    #         "content": chat["text"],
                    #         "alias": speaker_a,
                    #         "created_at": timestamp,
                    #     }
                    # )
                elif chat["speaker"] == speaker_b:
                    # messages.append(
                    #     {
                    #         "role": "assistant",
                    #         "content": chat["text"],
                    #         "alias": speaker_b,
                    #         "created_at": timestamp,
                    #     }
                    # )
                    messages_reverse.append(
                        {
                            "role": "user",
                            "content": chat["text"],
                            "alias": speaker_b,
                            "created_at": timestamp,
                        }
                    )
                else:
                    raise ValueError(f"Unknown speaker: {chat['speaker']}")

            # #### TODO: SAVE MESSAGES TO JSON FILES ####
            # output_dir = os.path.join(os.path.dirname(self.data_path), "processed_messages")
            # os.makedirs(output_dir, exist_ok=True)
            
            # messages_file = os.path.join(output_dir, f"messages_{idx}_{key}.json")
            # messages_reverse_file = os.path.join(output_dir, f"messages_reverse_{idx}_{key}.json")
            
            # with open(messages_file, 'w') as f:
            #     json.dump(messages, f, indent=2)
            # with open(messages_reverse_file, 'w') as f:
            #     json.dump(messages_reverse, f, indent=2)

            # add memories for the two users on different threads
            thread_a = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(
                    speaker_a_user_id,
                    messages,
                    "Adding Memories for Speaker A",
                ),
            )
            thread_b = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(
                    speaker_b_user_id,
                    messages_reverse,
                    "Adding Memories for Speaker B",
                ),
            )

            thread_a.start()
            thread_b.start()
            thread_a.join()
            thread_b.join()

        print("Messages added successfully")

    def process_all_conversations(self, max_workers=10, max_samples=None):
        if not self.data:
            raise ValueError(
                "No data loaded. Please set data_path and call load_data() first."
            )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if max_samples:
                self.data = self.data[:max_samples]
            futures = [
                executor.submit(self.process_conversation, item, idx)
                for idx, item in enumerate(self.data)
            ]

            for future in futures:
                future.result()
