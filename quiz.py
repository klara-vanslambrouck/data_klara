import streamlit as st
from quiz_data import quiz_data


def show_question():
    q_index = st.session_state.current_question
    question = quiz_data[q_index]

    st.write(f"Question {q_index + 1} of {len(quiz_data)}")
    st.write(question["question"])

    selected_choice = st.radio(
        "Vyber odpověď:",
        question["choices"],
        key=f"q_{q_index}"
    )

    if st.button("Odeslat", key=f"submit_{q_index}"):
        check_answer(selected_choice)


def check_answer(selected_choice):
    q_index = st.session_state.current_question
    question = quiz_data[q_index]

    if selected_choice == question["answer"]:
        st.success("Správně!")
        st.balloons()
        st.session_state.score += 1
    else:
        st.error(f"Špatná odpověď! Správná odpověď je: {question['answer']}.")

    if q_index + 1 < len(quiz_data):
        st.session_state.current_question += 1
    else:
        st.session_state.quiz_complete = True


def main():
    if "current_question" not in st.session_state:
        st.session_state.current_question = 0
    if "score" not in st.session_state:
        st.session_state.score = 0
    if "quiz_complete" not in st.session_state:
        st.session_state.quiz_complete = False

    st.title("Přátelé v datech - kvíz")

    if st.session_state.quiz_complete:
        st.success(
            f"Konec kvízu! Vaše skóre: "
            f"{st.session_state.score}/{len(quiz_data)}"
        )
        if st.button("Začít znovu"):
            st.session_state.current_question = 0
            st.session_state.score = 0
            st.session_state.quiz_complete = False
    else:
        show_question()


if __name__ == "__main__":
    main()
