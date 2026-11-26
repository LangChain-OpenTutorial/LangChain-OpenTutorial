def format_docs(docs):
    return "\n".join(
        [
            f"<document><content>{doc.page_content}</content><source>{doc.metadata['source']}</source><page>{int(doc.metadata['page'])+1}</page></document>"
            for doc in docs
        ]
    )


def format_searched_docs(docs):
    return "\n".join(
        [
            f"<document><content>{doc['content']}</content><source>{doc['url']}</source></document>"
            for doc in docs
        ]
    )


def format_task(tasks):
    # Create an empty list to store the results
    task_time_pairs = []

    # Traverse the list and process each item
    for item in tasks:
        # Separate strings by colon (:)
        task, time_str = item.rsplit(":", 1)
        # Remove the string 'time' and convert it to an integer
        time = int(time_str.replace("시간", "").strip())
        # Create to-dos and times as tuples and add them to a list
        task_time_pairs.append((task, time))

    # Output the results
    return task_time_pairs