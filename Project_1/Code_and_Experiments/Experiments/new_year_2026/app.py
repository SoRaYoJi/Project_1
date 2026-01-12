import streamlit as st
import random

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ComSci Bingo Spinner", page_icon="🔢")

st.title("🔢 ComSci Bingo Lucky Draw")
st.write("สุ่มเลขปิงโก 1-100 สำหรับกิจกรรมปีใหม่")

# --- การจัดการ State ---
# สร้างตัวแปรเก็บไว้ใน Session เพื่อไม่ให้ค่าหายเวลา Refresh หน้าจอ
if 'available_numbers' not in st.session_state:
    st.session_state.available_numbers = list(range(1, 101))
if 'drawn_numbers' not in st.session_state:
    st.session_state.drawn_numbers = []
if 'current_number' not in st.session_state:
    st.session_state.current_number = None

# --- ฟังก์ชันการทำงาน ---
def draw_number():
    if st.session_state.available_numbers:
        # สุ่มเลขจากรายการที่เหลืออยู่
        new_number = random.choice(st.session_state.available_numbers)
        # ย้ายจากรายการที่เหลือ ไปยังรายการที่ออกแล้ว
        st.session_state.available_numbers.remove(new_number)
        st.session_state.drawn_numbers.insert(0, new_number) # ใส่ไว้หน้าสุดของลิสต์
        st.session_state.current_number = new_number
    else:
        st.warning("สุ่มครบทุกเลขแล้ว (1-100)!")

def reset_game():
    st.session_state.available_numbers = list(range(1, 101))
    st.session_state.drawn_numbers = []
    st.session_state.current_number = None
    st.toast("รีเซ็ตเริ่มเกมใหม่แล้ว!")

# --- ส่วนการแสดงผล (UI) ---

col1, col2 = st.columns([1, 1])

with col1:
    if st.button('🎲 สุ่มเลขถัดไป', on_click=draw_number, use_container_width=True, type="primary"):
        pass

with col2:
    if st.button('🔄 เริ่มเกมใหม่ (Reset)', on_click=reset_game, use_container_width=True):
        pass

# แสดงเลขปัจจุบัน
st.divider()
if st.session_state.current_number:
    st.markdown(f"<h1 style='text-align: center; font-size: 100px; color: #FF4B4B;'>{st.session_state.current_number}</h1>", unsafe_allow_html=True)
else:
    st.markdown("<h1 style='text-align: center; font-size: 50px; color: gray;'>กดปุ่มเพื่อเริ่มสุ่ม</h1>", unsafe_allow_html=True)
st.divider()

# แสดงประวัติเลขที่ออกไปแล้ว
st.subheader(f"เลขที่ออกไปแล้ว ({len(st.session_state.drawn_numbers)}/100)")
if st.session_state.drawn_numbers:
    # แสดงเป็นแถบตัวเลขสวยๆ
    drawn_str = ", ".join(map(str, st.session_state.drawn_numbers))
    st.write(drawn_str)
else:
    st.info("ยังไม่มีเลขที่ถูกสุ่ม")

# แสดงจำนวนที่เหลือ
st.caption(f"เหลือตัวเลขในโถสุ่มอีก {len(st.session_state.available_numbers)} เลข")