"""SmartDesk AI - Intelligent Help Desk Management System.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from database import db
from ml import predictor
from ml.train_model import train
from services.classification import classify_complaint
from services.priority import detect_priority
from services.response import get_suggested_response
from services.validation import validate_complaint

st.set_page_config(
    page_title="SmartDesk AI",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

db.init_db()

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1200px; }
    .app-title { font-size: 2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.1rem; }
    .app-subtitle { color: #475569; margin-bottom: 1rem; }
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        margin-bottom: 8px;
    }
    .stat-label { color: #64748B; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; }
    .stat-value { font-size: 2rem; font-weight: 700; line-height: 1.2; }
    .section-note { color: #64748B; font-size: 0.9rem; }
    .ticket-box {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-left: 5px solid #2563EB;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0 18px 0;
    }
    .ticket-box h4 { margin: 0 0 4px 0; color: #1E3A8A; }
    .ticket-box p { margin: 2px 0; color: #334155; }
    div[data-testid="stSidebar"] { border-right: 1px solid #E2E8F0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def stat_card(label: str, value: object, color: str = "#1E3A8A") -> None:
    """Render one compact dashboard statistic card."""
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value" style="color: {color};">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ticket_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a clean, readable ticket table from database rows."""
    if frame.empty:
        return frame
    display = pd.DataFrame(
        {
            "Ticket ID": frame["id"].apply(db.format_ticket_id),
            "Subject": frame["subject"],
            "Category": frame["category"],
            "Priority": frame["priority"],
            "Status": frame["status"],
            "Created At": frame["created_at"],
        }
    )
    return display


def train_model_from_ui() -> None:
    """Train the model on demand and refresh the cached pipeline."""
    with st.spinner("Training the TF-IDF + Naive Bayes model..."):
        train()
        predictor.clear_model_cache()


def sidebar() -> str:
    """Draw the sidebar navigation and return the selected page."""
    with st.sidebar:
        st.markdown("## 🎫 SmartDesk AI")
        st.caption("Intelligent Help Desk Management System")
        st.divider()
        page = st.radio(
            "Navigation",
            ("Dashboard", "Submit Complaint", "Manage Tickets", "About Project"),
            label_visibility="collapsed",
        )
        st.divider()

        if predictor.model_exists():
            st.success("AI model: Ready")
        else:
            st.warning("AI model: Not trained")
            if st.button("Train model now", type="primary", width="stretch"):
                train_model_from_ui()
                st.rerun()

        st.caption("Local application · SQLite database · Scikit-learn")
    return page


def dashboard_page() -> None:
    st.markdown('<div class="app-title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Live ticket statistics read directly from the SQLite database.</div>',
        unsafe_allow_html=True,
    )

    stats = db.get_statistics()
    status_counts = stats["status_counts"]

    columns = st.columns(4)
    with columns[0]:
        stat_card("Total Tickets", stats["total"], "#1E3A8A")
    with columns[1]:
        stat_card("Pending", status_counts["Pending"], "#B45309")
    with columns[2]:
        stat_card("In Progress", status_counts["In Progress"], "#1D4ED8")
    with columns[3]:
        stat_card("Resolved", status_counts["Resolved"], "#047857")

    st.write("")
    chart_columns = st.columns(3)

    with chart_columns[0]:
        st.subheader("Tickets by Category")
        if stats["category_counts"]:
            category_frame = pd.DataFrame(
                {"Tickets": stats["category_counts"]}
            )
            st.bar_chart(category_frame)
        else:
            st.info("No category data yet.")

    with chart_columns[1]:
        st.subheader("Tickets by Priority")
        if stats["priority_counts"]:
            priority_frame = pd.DataFrame({"Tickets": stats["priority_counts"]})
            st.bar_chart(priority_frame)
        else:
            st.info("No priority data yet.")

    with chart_columns[2]:
        st.subheader("Tickets by Status")
        status_frame = pd.DataFrame(
            {
                "Tickets": [
                    status_counts["Pending"],
                    status_counts["In Progress"],
                    status_counts["Resolved"],
                ]
            },
            index=["Pending", "In Progress", "Resolved"],
        )
        st.bar_chart(status_frame)

    st.divider()
    st.subheader("Recent Tickets")
    recent = db.get_all_tickets()
    if recent.empty:
        st.info("No tickets have been submitted yet. Use the Submit Complaint page to create one.")
    else:
        st.dataframe(
            ticket_table(recent.head(5)),
            hide_index=True,
            width="stretch",
        )
        st.caption(f"Showing 5 of {len(recent)} ticket(s). Manage all tickets from the Manage Tickets page.")


def submit_page() -> None:
    st.markdown('<div class="app-title">Submit a Complaint</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Describe the issue. The trained AI model will classify it and the system will assign a priority and first-line response.</div>',
        unsafe_allow_html=True,
    )

    if not predictor.model_exists():
        st.error(
            "The AI model is not trained yet. Train it from the sidebar or run "
            "`python -m ml.train_model`, then reload this page."
        )
        return

    with st.form("complaint_form", clear_on_submit=False):
        row = st.columns(2)
        with row[0]:
            name = st.text_input("Full Name", placeholder="Enter your full name")
        with row[1]:
            email = st.text_input("Email Address", placeholder="you@example.com")
        subject = st.text_input("Complaint Subject", placeholder="Short summary of the issue")
        description = st.text_area(
            "Complaint Description",
            placeholder="Explain the issue in detail. Include error messages or symptoms if possible.",
            height=160,
        )
        submitted = st.form_submit_button(
            "Submit Complaint", type="primary", width="stretch"
        )

    if submitted:
        errors = validate_complaint(name, email, subject, description)
        if errors:
            st.error("Please correct the following before submitting:")
            for field, message in errors.items():
                st.warning(message)
        else:
            with st.spinner("Running AI classification..."):
                classification = classify_complaint(subject, description)
                priority_result = detect_priority(f"{subject} {description}")
                suggested_response = get_suggested_response(
                    str(classification["category"])
                )
                ticket_id = db.create_ticket(
                    name=name,
                    email=email,
                    subject=subject,
                    description=description,
                    category=str(classification["category"]),
                    priority=str(priority_result["priority"]),
                    suggested_response=suggested_response,
                    confidence=float(classification["confidence"]),
                )

            st.session_state["last_ticket"] = {
                "id": ticket_id,
                "category": classification["category"],
                "confidence": classification["confidence"],
                "low_confidence": classification["low_confidence"],
                "note": classification["note"],
                "priority": priority_result["priority"],
                "priority_reason": priority_result["reason"],
                "response": suggested_response,
            }

    last_ticket = st.session_state.get("last_ticket")
    if last_ticket:
        st.success(f"Complaint submitted successfully. Your ticket ID is {db.format_ticket_id(last_ticket['id'])}.")
        st.markdown(
            f"""
            <div class="ticket-box">
                <h4>{db.format_ticket_id(last_ticket['id'])}</h4>
                <p><strong>Status:</strong> Pending</p>
                <p><strong>Predicted Category:</strong> {last_ticket['category']}</p>
                <p><strong>Prediction Confidence:</strong> {last_ticket['confidence'] * 100:.1f}%</p>
                <p><strong>Priority:</strong> {last_ticket['priority']}</p>
                <p><strong>Priority Rule:</strong> {last_ticket['priority_reason']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if last_ticket["low_confidence"]:
            st.warning(str(last_ticket["note"]))
        st.info(f"**Suggested Response**\n\n{last_ticket['response']}")
        st.caption(
            "The suggested response is an automated first-line reply. An administrator "
            "will review the ticket and update its status."
        )


def manage_page() -> None:
    st.markdown('<div class="app-title">Manage Tickets</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Search, filter, inspect and update any complaint stored in the database.</div>',
        unsafe_allow_html=True,
    )

    flash = st.session_state.pop("manage_flash", None)
    if flash:
        st.success(flash)

    filter_columns = st.columns([2, 1, 1, 1])
    with filter_columns[0]:
        search = st.text_input(
            "Search by Ticket ID or Subject",
            placeholder="Example: TKT-000001 or printer",
        )
    with filter_columns[1]:
        category = st.selectbox("Category", ("All",) + db.CATEGORIES)
    with filter_columns[2]:
        priority = st.selectbox("Priority", ("All",) + db.PRIORITIES)
    with filter_columns[3]:
        status = st.selectbox("Status", ("All",) + db.STATUSES)

    tickets = db.get_all_tickets(
        search=search,
        category=category,
        priority=priority,
        status=status,
    )

    st.caption(f"{len(tickets)} ticket(s) match the current filters.")
    if tickets.empty:
        st.info("No tickets match the selected filters.")
        return

    st.dataframe(ticket_table(tickets), hide_index=True, width="stretch")
    st.divider()

    st.subheader("Ticket Details")
    selected_id = st.selectbox(
        "Select a ticket",
        options=tickets["id"].tolist(),
        format_func=db.format_ticket_id,
    )
    ticket = db.get_ticket(int(selected_id))
    if ticket is None:
        st.error("The selected ticket could not be found.")
        return

    st.markdown(
        f"### {db.format_ticket_id(ticket['id'])} — {ticket['subject']}"
    )

    info_columns = st.columns(4)
    with info_columns[0]:
        st.markdown(f"**Status**  \n{ticket['status']}")
    with info_columns[1]:
        st.markdown(f"**Priority**  \n{ticket['priority']}")
    with info_columns[2]:
        st.markdown(f"**AI Category**  \n{ticket['category']}")
    with info_columns[3]:
        st.markdown(f"**Confidence**  \n{ticket['confidence'] * 100:.1f}%")

    detail_columns = st.columns(2)
    with detail_columns[0]:
        st.markdown(f"**Submitted By**  \n{ticket['name']}")
        st.markdown(f"**Email**  \n{ticket['email']}")
    with detail_columns[1]:
        st.markdown(f"**Created At**  \n{ticket['created_at']}")
        st.markdown(f"**Suggested Response**")

    st.write("**Complaint Description**")
    st.write(ticket["description"])
    st.info(ticket["suggested_response"])

    if ticket["confidence"] < 0.30:
        st.warning(
            "This prediction had low confidence and was routed to General Inquiry "
            "for manual review."
        )

    st.write("")
    st.subheader("Update Status")
    update_columns = st.columns([2, 1])
    with update_columns[0]:
        new_status = st.selectbox(
            "New status",
            db.STATUSES,
            index=db.STATUSES.index(ticket["status"]),
            key=f"status_{ticket['id']}",
        )
    with update_columns[1]:
        st.write("")
        st.write("")
        if st.button("Update Status", type="primary", width="stretch"):
            if db.update_ticket_status(int(ticket["id"]), new_status):
                st.session_state["manage_flash"] = (
                    f"{db.format_ticket_id(ticket['id'])} status updated to {new_status}."
                )
                st.rerun()
            else:
                st.error("The ticket could not be updated.")


def about_page() -> None:
    st.markdown('<div class="app-title">About Project</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">SmartDesk AI — Third Year Engineering Mini-Project (Fundamentals of AI Lab)</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Project Overview")
        st.write(
            "SmartDesk AI is a local help desk application that accepts employee "
            "complaints, predicts their category with a trained Machine Learning "
            "model, assigns an explainable priority, suggests a first-line response, "
            "and stores every ticket in SQLite. Administrators can search, filter and "
            "update tickets from the Manage Tickets page."
        )

        st.subheader("Problem Statement")
        st.write(
            "Manual complaint sorting is slow and inconsistent. Support staff must read "
            "each complaint, decide its category and priority, and write a first response. "
            "SmartDesk AI automates the first pass so that tickets reach the right team faster."
        )

        st.subheader("Objectives")
        st.markdown(
            """
            - Accept and validate user complaints through a web form.
            - Classify complaint text using a real Machine Learning model.
            - Show prediction confidence instead of hiding uncertainty.
            - Detect priority with transparent rule-based logic.
            - Suggest a standard response based on the predicted category.
            - Store tickets permanently in SQLite.
            - Give administrators a live dashboard and ticket management page.
            """
        )

    with right:
        st.subheader("Technology Stack")
        st.markdown(
            """
            | Layer | Technology |
            |---|---|
            | Language | Python |
            | Frontend | Streamlit |
            | ML library | Scikit-learn |
            | Features | TF-IDF |
            | Classifier | Multinomial Naive Bayes |
            | Database | SQLite |
            | Data processing | Pandas |
            | Testing | Pytest |
            | Model storage | Joblib |
            """
        )
        st.subheader("How the AI Works")
        st.markdown(
            """
            1. Complaint text is cleaned and split into words and word pairs.
            2. TF-IDF converts those terms into numerical weights. Words that are
               common in one category but rare elsewhere get higher weight.
            3. Multinomial Naive Bayes learns the probability of each term for each
               category during training.
            4. For a new complaint, the model combines those probabilities and returns
               the most likely category with a confidence score.
            """
        )

    st.divider()
    st.subheader("Model Evaluation (Real Results)")
    metrics = predictor.get_model_metrics()
    if metrics is None:
        st.warning(
            "No evaluation results found. Train the model with "
            "`python -m ml.train_model` and reload this page."
        )
        return

    metric_columns = st.columns(4)
    with metric_columns[0]:
        stat_card("Accuracy", f"{metrics['accuracy'] * 100:.2f}%", "#1E3A8A")
    with metric_columns[1]:
        stat_card("Precision (macro)", f"{metrics['precision_macro'] * 100:.2f}%", "#1D4ED8")
    with metric_columns[2]:
        stat_card("Recall (macro)", f"{metrics['recall_macro'] * 100:.2f}%", "#047857")
    with metric_columns[3]:
        stat_card("F1-score (macro)", f"{metrics['f1_macro'] * 100:.2f}%", "#B45309")

    st.caption(
        f"Trained on {metrics['train_size']} records and evaluated on "
        f"{metrics['test_size']} held-out records (dataset size: {metrics['dataset_size']}, "
        f"random state: {metrics['random_state']}, trained at: {metrics['trained_at']})."
    )

    report = metrics["classification_report"]
    labels = metrics["labels"]
    rows = []
    for label in labels:
        label_metrics = report[label]
        rows.append(
            {
                "Category": label,
                "Precision": round(label_metrics["precision"], 4),
                "Recall": round(label_metrics["recall"], 4),
                "F1-score": round(label_metrics["f1-score"], 4),
                "Support": int(label_metrics["support"]),
            }
        )
    st.write("**Per-category evaluation on the test set**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with st.expander("Confusion matrix and model details"):
        matrix_frame = pd.DataFrame(
            metrics["confusion_matrix"], index=labels, columns=labels
        )
        st.write("Rows are actual categories; columns are predicted categories.")
        st.dataframe(matrix_frame, width="stretch")
        st.json(
            {
                "model": metrics["model"],
                "alpha": metrics["alpha"],
                "test_split": metrics["test_split"],
                "random_state": metrics["random_state"],
                "dataset_size": metrics["dataset_size"],
            }
        )

    st.info(
        "This project uses a small demonstration dataset (100 sample complaints). "
        "The metrics above are real results from a single train/test split, but they "
        "do not prove performance on real-world complaints. A production system would "
        "need more representative, balanced and regularly updated data."
    )

    if st.button("Retrain Model Now", type="primary"):
        train_model_from_ui()
        st.success("Model retrained successfully. Metrics on this page have been refreshed.")
        st.rerun()


def main() -> None:
    page = sidebar()

    if page == "Dashboard":
        dashboard_page()
    elif page == "Submit Complaint":
        submit_page()
    elif page == "Manage Tickets":
        manage_page()
    else:
        about_page()


if __name__ == "__main__":
    main()
