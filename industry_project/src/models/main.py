from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Import your working Bayesian model functions!
from bayesian import train_bayesian_model, get_bayesian_prediction

# ==========================================
# 1. THE STARTUP MANAGER
# ==========================================
# This tells FastAPI to load the CSVs and train the model ONCE when it boots up.
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⚙️ API Server starting up: Initializing the Zeiss Quality Gate...")
    success = train_bayesian_model()
    if not success:
        print("⚠️ CRITICAL: Model failed to train on startup. Check the data folder.")
    yield
    print("🛑 API Server shutting down.")

# Initialize the actual API application
app = FastAPI(
    title="Zeiss Predictive Reliability Engine", 
    description="AI-Driven Quality Gate Prototype",
    version="1.0",
    lifespan=lifespan
)

# ==========================================
# 2. INPUT VALIDATION
# ==========================================
# This strictly enforces what data Azure DevOps is allowed to send to the API.
class BuildMetrics(BaseModel):
    code_churn: int
    crit_vulns: int
    test_pass_rate: float

# ==========================================
# 3. THE ENDPOINTS
# ==========================================

@app.get("/")
def health_check():
    """A simple ping to check if the server is alive."""
    return {"status": "Operational", "message": "Zeiss Quality Gate API is running."}

@app.post("/predict")
def predict_build_risk(metrics: BuildMetrics):
    """
    Receives build metrics and returns a Confidence Score.
    This is the endpoint the Azure CI/CD pipeline will talk to.
    """
    # 1. Pass the incoming numbers to your trained model
    score = get_bayesian_prediction(
        code_churn=metrics.code_churn, 
        crit_vulns=metrics.crit_vulns, 
        test_pass_rate=metrics.test_pass_rate
    )
    
    # 2. Handle errors if the model isn't ready
    if score == -1.0:
        raise HTTPException(
            status_code=503, 
            detail="Model is currently training or missing data. Try again in a few seconds."
        )
        
    # 3. Calculate the Risk Level and Confidence
    # A crash probability of 91.7% means the "Confidence" in the build is only 8.3%
    confidence_score = 1.0 - score 
    risk_level = "High" if score > 0.50 else "Low"
        
    # 4. Return the official JSON response
    return {
        "Received_Metrics": metrics,
        "Crash_Probability": round(score, 3),
        "Confidence_Score": round(confidence_score, 3),
        "Risk_Level": risk_level,
        "Recommendation": "Block Release" if risk_level == "High" else "Approve Release"
    }