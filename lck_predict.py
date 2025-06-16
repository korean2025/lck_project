import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, flash, session
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from functools import wraps


# 데이터 로드 및 전처리
BASE_PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
lck = pd.read_csv(os.path.join(BASE_PROJECT_PATH, 'lck_num.csv'))

# lck = pd.read_csv('C:/Users/User/Desktop/나익준/addSimulator1_add_pyfile/lck_num.csv') ####경로 지정

lck['GRB%'] = lck['GRB%'].fillna(0)
lck['ELD%'] = lck['ELD%'].fillna(0)
lck['BN%'] = lck['BN%'].fillna(0)

# 'K', 'D', 'PPG' 컬럼 숫자 변환 및 NaN 처리
lck['K'] = pd.to_numeric(lck['K'], errors='coerce')
lck['K'] = lck['K'].fillna(lck['K'].mean())
lck['D'] = pd.to_numeric(lck['D'], errors='coerce')
lck['D'] = lck['D'].fillna(lck['D'].mean())
lck['PPG'] = pd.to_numeric(lck['PPG'], errors='coerce')
lck['PPG'] = lck['PPG'].fillna(0)

# Label Encoding 적용
lb = LabelEncoder()
for col in ['Team', 'GSPD', 'FB%', 'FT%', 'F3T%', 'HLD%', 'GRB%', 'FD%', 'DRG%', 'ELD%', 'FBN%', 'BN%', 'LNE%', 'JNG%']:
    lck[col] = lb.fit_transform(lck[col])

# 추가 NaN 값 처리
lck['KD'] = lck['KD'].fillna(lck['KD'].mean())
lck['CKPM'] = lck['CKPM'].fillna(lck['CKPM'].mean())
lck['GPR'] = lck['GPR'].fillna(lck['GPR'].mean())
lck['EGR'] = lck['EGR'].fillna(lck['EGR'].mean())
lck['MLR'] = lck['MLR'].fillna(lck['MLR'].mean())
lck['GD15'] = lck['GD15'].fillna(lck['GD15'].mean())
lck['WPM'] = lck['WPM'].fillna(lck['WPM'].mean())
lck['CWPM'] = lck['CWPM'].fillna(lck['CWPM'].mean())
lck['WCPM'] = lck['WCPM'].fillna(lck['WCPM'].mean())

# 종속 변수 생성 및 데이터 분리
lck['GP_W'] = lck['W'] / lck['GP']
y_data = lck['GP_W']
x_data = lck.drop(['GP_W'], axis=1)

# 훈련 및 테스트 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=42, test_size=0.2)

# GradientBoostingRegressor 모델 학습
gr = GradientBoostingRegressor(random_state=42, max_depth=20)
gr.fit(x_train, y_train)

# 팀 이름과 인코딩된 값 매핑 (하드코딩된 값 사용)
teams = {
    "BNK FEARX": 0, "BRION": 1, "DN Freecs": 2, "DRX": 3, "Dplus KIA": 4,
    "Gen.G": 5, "Hanwha Life Esports": 6, "KT Rolster": 7, "Liiv SANDBOX": 8,
    "Nongshim RedForce": 9, "OKSavingsBank BRION": 10, "T1": 11
}

# --- 중요한 변경: 프로젝트 기본 경로를 명시적으로 설정 ---
# 이 경로를 여러분의 'project_lck_250605' 폴더의 실제 절대 경로로 직접 지정해주세요.
# 예: "C:/Users/PHANTOM/Desktop/a9a9a9/project_lck_250605"

# BASE_PROJECT_PATH = "C:/Users/User/Desktop/나익준/addSimulator1_add_pyfile"  # 변환된 경로를 프로그램 변수에 저장
import pymysql
# 현재 파일의 디렉토리를 프로젝트 기본 경로로 설정
BASE_PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))

# Flask 애플리케이션 초기화
app = Flask(__name__,
            template_folder=os.path.join(BASE_PROJECT_PATH, "templates"),
            static_folder=os.path.join(BASE_PROJECT_PATH, "static"))


#pip install pymysql

pymysql.install_as_MySQLdb()

# SocketIO 초기화에 필요한 SECRET_KEY 설정
app.config['SECRET_KEY'] = 'your_secret_key_for_socketio'

# --- 데이터베이스 설정 추가 ---
# site.db 파일 경로를 BASE_PROJECT_PATH를 기반으로 설정

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:1234@localhost/my_database'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app) # SQLAlchemy 객체 초기화

# --- 데이터베이스 모델 정의 ---
class Post(db.Model):
    __tablename__ = 'Post'  # 변경: 테이블 이름을 명시적으로 설정
    id = db.Column(db.Integer, primary_key=True, default=0)
    title = db.Column(db.String(100), nullable=False, default="제목 없음")
    content = db.Column(db.Text, nullable=False, default="내용 없음")
    author = db.Column(db.String(20), nullable=False, default='익명')
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"Post('{self.title}', '{self.date_posted}')"

class Comment(db.Model):
    __tablename__ = 'Comment'  # 변경: 테이블 이름을 명시적으로 설정
    id = db.Column(db.Integer, primary_key=True, default=0)
    post_id = db.Column(db.Integer, db.ForeignKey('Post.id', ondelete='CASCADE'), nullable=False, default=0)  # 변경: 'Post.id'로 지정
    content = db.Column(db.Text, nullable=False, default="내용 없음")
    author = db.Column(db.String(20), nullable=False, default="익명")
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    post = db.relationship('Post', backref=db.backref('comments', lazy=True, cascade="all, delete-orphan"))

    def __repr__(self):
        return f"Comment('{self.content}', '{self.date_posted}')"

# --- 새로운 Notice 모델 추가 ---
class Notice(db.Model):
    id = db.Column(db.Integer, primary_key=True, default=0)
    title = db.Column(db.String(200), nullable=False, default="제목 없음")
    content = db.Column(db.Text, nullable=False, default="내용 없음")
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"Notice('{self.title}', '{self.date_posted}')"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    team_name = data.get('team')
    
    if team_name not in teams:
        return jsonify({"error": "잘못된 팀 이름입니다."})

    team_encoded = teams[team_name]
    # 예측을 위한 x_train 필터링 로직 수정 (적절한 특성만 포함되도록)
    # 현재 x_train의 모든 컬럼을 사용하고 있으므로, 예측 시에도 동일한 컬럼 구조를 유지해야 합니다.
    # 단일 팀의 예측을 위해 해당 팀의 데이터를 x_train에서 필터링하여 사용합니다.
    # 실제 환경에서는 예측에 필요한 모든 특성(feature)을 클라이언트로부터 받거나,
    # 예측을 위한 더미 데이터를 생성하여 모델의 입력 차원과 일치시켜야 합니다.
    # 여기서는 예시로 'Team' 컬럼만으로 필터링하지만, 실제 모델 입력은 전체 특성을 필요로 합니다.
    # 따라서 이 부분은 실제 모델의 입력 요구사항에 맞게 조정해야 합니다.
    
    # 임시 방편으로, 해당 팀의 x_data를 가져와서 예측합니다.
    # 이는 해당 팀의 기존 데이터셋 내의 모든 경기에 대한 평균 승률을 반환할 것입니다.
    # 만약 특정 경기 또는 특정 상태의 예측을 원한다면, 해당 상태에 맞는 새로운 데이터를 구성해야 합니다.
    
    # 주의: 이 예측 방식은 단순히 해당 팀의 과거 데이터 기반 평균 승률을 반환합니다.
    # 실제 시뮬레이션에서는 시뮬레이션할 특정 경기 상황의 모든 특징을 입력해야 합니다.
    team_data_for_prediction = x_data[x_data['Team'] == team_encoded]
    
    if team_data_for_prediction.empty:
        return jsonify({"error": f"팀 {team_name}에 대한 데이터가 없습니다."})

    prediction = gr.predict(team_data_for_prediction).mean() * 100

    print("팀 숫자 변환 값 :", team_encoded)

    return jsonify({
        "team_name": team_name,
        "win_rate": round(prediction, 2)
    })

# --- SocketIO 초기화 ---
socketio = SocketIO(app)

# --- 라우트 정의 ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Simulator')
def Simulator():
    return render_template('Simulator.html')

@app.route('/bnk_fearx')
def bnk_fearx():
    return render_template('BNK FEARX.html')

@app.route('/dn_freecs')
def dn_freecs():
    return render_template('DN Freecs.html')

@app.route('/dplus_kia')
def dplus_kia():
    return render_template('Dplus KIA.html')

@app.route('/drx')
def drx():
    return render_template('DRX.html')

@app.route('/gen_g')
def gen_g():
    return render_template('Gen.G.html')

@app.route('/hanwha_life_esports')
def hanwha_life_esports():
    return render_template('Hanwha Life Esports.html')

@app.route('/kt_rolster')
def kt_rolster():
    return render_template('KT Rolster.html')

@app.route('/nongshim_redforce')
def nongshim_redforce():
    return render_template('Nongshim RedForce.html')

@app.route('/oksavingsbank_brion')
def okSavingsBank_brion():
    return render_template('BRION.html')

@app.route('/player_comparison')
def player_comparison():
    return render_template('Player comparison.html')

@app.route('/T1')
def T1():
    return render_template('T1.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# --- SocketIO 이벤트 핸들러 ---
@app.route('/chat')
def chat():
    return render_template('chat.html')

@socketio.on('connect')
def test_connect():
    print('Client connected')
    emit('status', {'msg': '새로운 사용자가 입장했습니다.'}, broadcast=True)

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')
    emit('status', {'msg': '사용자가 퇴장했습니다.'}, broadcast=True)

@socketio.on('message')
def handle_message(data):
    print('received message: ' + str(data))
    emit('message', data, broadcast=True)

# --- 게시판 관련 라우트 ---

# @app.route('/board')
# def board():
#     posts = Post.query.order_by(Post.date_posted.desc()).all()
#     return render_template('board.html', posts=posts)

@app.route('/board')
def board():
    # 일반 게시글 데이터를 Post 테이블에서 가져오기
    posts = Post.query.order_by(Post.date_posted.desc()).all()
    # 공지사항 데이터를 Notice 테이블에서 가져오기 (추가)
    notices = Notice.query.order_by(Notice.date_posted.desc()).all()
    # `notices`와 `posts`를 템플릿으로 전달
    return render_template('board.html', posts=posts, notices=notices)

@app.route('/board/new', methods=['GET', 'POST'])
def new_post():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        author = request.form.get('author', '익명')

        post = Post(title=title, content=content, author=author)
        db.session.add(post)
        db.session.commit()
        return redirect(url_for('board'))
    return render_template('create_post.html')

@app.route('/board/<int:post_id>')
def post_detail(post_id):
    post = Post.query.get_or_404(post_id)
    return render_template('post_detail.html', post=post)

@app.route('/board/<int:post_id>/update', methods=['GET', 'POST'])
def update_post(post_id):
    post = Post.query.get_or_404(post_id)
    if request.method == 'POST':
        post.title = request.form['title']
        post.content = request.form['content']
        post.author = request.form.get('author', '익명')
        db.session.commit()
        return redirect(url_for('post_detail', post_id=post.id))
    return render_template('update_post.html', post=post)

@app.route('/board/<int:post_id>/delete', methods=['POST'])
def delete_post(post_id):
    post = Post.query.get_or_404(post_id)
    Comment.query.filter_by(post_id=post_id).delete()
    db.session.delete(post)
    db.session.commit()
    flash('게시글이 삭제되었습니다.', 'success')
    return redirect(url_for('board'))

@app.route('/board/<int:post_id>/comment', methods=['POST'])
def add_comment(post_id):
    post = Post.query.get_or_404(post_id)
    content = request.form['content']
    author = request.form.get('author', '익명')

    comment = Comment(post_id=post.id, content=content, author=author)
    db.session.add(comment)
    db.session.commit()

    return redirect(url_for('post_detail', post_id=post.id))
    
# --- 관리자 계정 정보 (하드코딩) ---
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "1234"

# --- 관리자 로그인 데코레이터 ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session or not session['admin_logged_in']:
            flash('관리자 권한이 필요합니다.', 'error')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function
    
# --- 관리자 로그인 및 로그아웃 ---
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            flash('관리자로 로그인되었습니다.', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('잘못된 아이디 또는 비밀번호입니다.', 'error')
    return render_template('admin_login.html')

@app.route('/admin/logout')
@admin_required
def admin_logout():
    session.pop('admin_logged_in', None)
    flash('관리자 로그아웃되었습니다.', 'info')
    return redirect(url_for('admin_login'))

# --- 관리자 대시보드 ---
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    posts = Post.query.order_by(Post.date_posted.desc()).all()
    # 공지사항 데이터 로드 추가
    notices = Notice.query.order_by(Notice.date_posted.desc()).all()
    return render_template('admin_dashboard.html', posts=posts, notices=notices)

# --- 관리자 게시물 삭제 라우트 ---
@app.route('/admin/posts/<int:post_id>', methods=['POST'])
@admin_required
def admin_delete_post(post_id):
    post = Post.query.get_or_404(post_id)
    
    # 게시물에 달린 모든 댓글 먼저 삭제
    Comment.query.filter_by(post_id=post_id).delete()
    db.session.delete(post)
    db.session.commit()
    flash('게시글 및 관련 댓글이 삭제되었습니다.', 'success')
    return redirect(url_for('admin_dashboard'))

# --- 게시물 상세 정보 및 댓글 불러오기 (API) ---
@app.route('/api/admin/posts/<int:post_id>')
@admin_required
def api_admin_post_detail(post_id):
    post = Post.query.get_or_404(post_id)
    comments = Comment.query.filter_by(post_id=post_id).order_by(Comment.date_posted.asc()).all()

    post_data = {
        'id': post.id,
        'title': post.title,
        'content': post.content,
        'author': post.author,
        'date_posted': post.date_posted.strftime('%Y-%m-%d %H:%M'),
    }
    comments_data = []
    for comment in comments:
        comments_data.append({
            'id': comment.id,
            'content': comment.content,
            'author': comment.author,
            'date_posted': comment.date_posted.strftime('%Y-%m-%d %H:%M')
        })
    post_data['comments'] = comments_data
    return jsonify(post_data)

# --- 관리자 댓글 삭제 (API) ---
@app.route('/api/admin/comments/<int:comment_id>/delete', methods=['POST'])
@admin_required
def api_admin_delete_comment(comment_id):
    comment = Comment.query.get(comment_id)
    if comment:
        db.session.delete(comment)
        db.session.commit()
        return jsonify({'success': '댓글이 삭제되었습니다.'})
    return jsonify({'error': '댓글을 찾을 수 없습니다.'}), 404

# --- 새로운 공지사항 추가 라우터 ---
@app.route('/admin/add_notice', methods=['POST'])
@admin_required
def admin_add_notice():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')

        if not title or not content:
            flash('제목과 내용을 모두 입력해주세요.', 'error')
            return redirect(url_for('admin_dashboard'))

        try:
            new_notice = Notice(title=title, content=content)
            db.session.add(new_notice)
            db.session.commit()
            flash('공지사항이 성공적으로 추가되었습니다.', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'공지사항 추가 중 오류가 발생했습니다: {e}', 'error')
        
        return redirect(url_for('admin_dashboard'))
    return redirect(url_for('admin_dashboard')) # POST 요청이 아니면 대시보드로 리다이렉트

# --- 새로운 공지사항 삭제 라우터 ---
@app.route('/admin/notices/<int:notice_id>/delete', methods=['POST'])
@admin_required
def admin_delete_notice(notice_id):
    notice = Notice.query.get_or_404(notice_id)
    try:
        db.session.delete(notice)
        db.session.commit()
        flash('공지사항이 성공적으로 삭제되었습니다.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'공지사항 삭제 중 오류가 발생했습니다: {e}', 'error')
    
    return redirect(url_for('admin_dashboard'))

@app.route('/api/admin/notices/<int:notice_id>', methods=['GET'])
@admin_required # 관리자 로그인 확인 데코레이터 (필요시)
def get_admin_notice_details(notice_id):
    notice = Notice.query.get(notice_id)
    if not notice:
        return jsonify({'error': '공지사항을 찾을 수 없습니다.'}), 404
    return jsonify({
        'id': notice.id,
        'title': notice.title,
        'content': notice.content,
        'date_posted': notice.date_posted.strftime('%Y-%m-%d %H:%M')
    })

# --- 애플리케이션 실행 ---
if __name__ == '__main__':
    # Flask 앱 컨텍스트 내에서 데이터베이스 테이블을 생성합니다.
    with app.app_context():
        db.create_all() # 데이터베이스 테이블 생성 (최초 1회만 실행)

    # Socket.IO를 포함하여 Flask 서버를 실행합니다.
    socketio.run(app, debug=True, use_reloader=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)