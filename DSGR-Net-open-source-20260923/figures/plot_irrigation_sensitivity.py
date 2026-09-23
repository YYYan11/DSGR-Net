import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ==================================================
# Path
# ==================================================

CSV_PATH = "figures/irrigation_response_final.csv"

SENS_PATH = "figures/daily_irrigation_sensitivity.npy"


OUT_DIR = Path("figures/generated")
OUT_DIR.mkdir(parents=True, exist_ok=True)


OUT_PNG = OUT_DIR / "Fig5_irrigation_gradient_analysis.png"
OUT_PDF = OUT_DIR / "Fig5_irrigation_gradient_analysis.pdf"

OUT_CSV = OUT_DIR / "Fig5_stage_sensitivity_summary.csv"



# ==================================================
# Load response data
# ==================================================

df = pd.read_csv(CSV_PATH)


dap = df["DAP"].values.astype(float)

response = df["median"].values.astype(float)

q25 = df["q25"].values.astype(float)

q75 = df["q75"].values.astype(float)



# ==================================================
# Load gradient sensitivity
# ==================================================

sens = np.load(SENS_PATH)


if sens.ndim == 1:

    sens_curve = np.abs(sens)


elif sens.ndim == 2:

    if sens.shape[1] == len(dap):

        sens_curve = np.mean(
            np.abs(sens),
            axis=0
        )

    elif sens.shape[0] == len(dap):

        sens_curve = np.mean(
            np.abs(sens),
            axis=1
        )

    else:

        raise ValueError(
            f"Unexpected sensitivity shape {sens.shape}"
        )


else:

    raise ValueError(
        f"Unsupported sensitivity dimension {sens.ndim}"
    )



# interpolation

if len(sens_curve) != len(dap):

    x_old = np.linspace(
        dap.min(),
        dap.max(),
        len(sens_curve)
    )

    sens_curve = np.interp(
        dap,
        x_old,
        sens_curve
    )




# ==================================================
# Growth stage definition
# ==================================================

stages = [

    ("Vegetative",0,55),

    ("Flowering",56,105),

    ("Boll filling",106,160),

    ("Maturity",161,int(dap.max()))

]


stage_colors = [

    "#66c2a5",

    "#fc8d62",

    "#ffd92f",

    "#8da0cb"

]



summary=[]



for name,start,end in stages:


    mask = (
        (dap>=start)
        &
        (dap<=end)
    )


    values=sens_curve[mask]

    days=dap[mask]


    mean_value=float(
        np.mean(values)
    )


    peak_index=int(
        np.argmax(values)
    )


    peak_value=float(
        values[peak_index]
    )


    peak_day=int(
        days[peak_index]
    )


    summary.append(
        [
            name,
            start,
            end,
            mean_value,
            peak_value,
            peak_day
        ]
    )



summary_df=pd.DataFrame(

    summary,

    columns=[
        "Stage",
        "Start_DAP",
        "End_DAP",
        "Mean_sensitivity",
        "Peak_sensitivity",
        "Peak_DAP"
    ]

)


summary_df.to_csv(
    OUT_CSV,
    index=False
)



# ==================================================
# Figure style
# ==================================================

plt.rcParams.update({

    "font.family":"sans-serif",

    "font.sans-serif":["Arial", "Helvetica", "DejaVu Sans"],

    "font.size":10,

    "axes.labelsize":11,

    "axes.titlesize":10,

    "xtick.labelsize":9,

    "ytick.labelsize":9,

    "pdf.fonttype":42

})



fig = plt.figure(

    figsize=(7.0,4.8)

)



gs = fig.add_gridspec(

    2,

    1,

    height_ratios=[3.0,2.0],

    hspace=0.45

)



# ==================================================
# (a) Temporal response
# ==================================================

ax1 = fig.add_subplot(gs[0])



ax1.fill_between(

    dap,

    q25,

    q75,

    color="#9ecae1",

    alpha=0.45,

    linewidth=0

)



ax1.plot(

    dap,

    response,

    color="#2171b5",

    linewidth=2.2

)



# peak response

peak_resp_idx=int(
    np.argmax(response)
)


peak_resp_dap=int(
    dap[peak_resp_idx]
)


peak_resp=response[peak_resp_idx]



ax1.scatter(

    peak_resp_dap,

    peak_resp,

    color="black",

    s=25,

    zorder=5

)



ax1.annotate(

    f"Peak response\nDAP={peak_resp_dap}",

    xy=(

        peak_resp_dap,

        peak_resp

    ),

    xytext=(

        35,

        -30

    ),

    textcoords="offset points",

    fontsize=8,

    arrowprops={

        "arrowstyle":"->",

        "lw":0.8

    }

)



ax1.set_title(

    "(a) Temporal irrigation response",

    pad=4,

    fontsize=10

)


ax1.set_ylabel(

    "Response magnitude"

)



ax1.grid(

    axis="y",

    linestyle="--",

    alpha=0.3

)



ax1.spines["top"].set_visible(False)

ax1.spines["right"].set_visible(False)



ax1.tick_params(

    axis="x",

    labelbottom=False

)



# ==================================================
# (b) Stage sensitivity
# ==================================================

ax2=fig.add_subplot(gs[1])



x=np.arange(
    len(summary_df)
)


mean_values=summary_df[
    "Mean_sensitivity"
].values



peak_values=summary_df[
    "Peak_sensitivity"
].values



peak_days=summary_df[
    "Peak_DAP"
].values




bars=ax2.bar(

    x,

    mean_values,

    width=0.65,

    color=stage_colors,

    edgecolor="black",

    linewidth=0.6,

    alpha=0.9,

    label="Mean sensitivity"

)




ax2.plot(

    x,

    peak_values,

    color="black",

    marker="o",

    linewidth=1.2,

    markersize=5,

    label="Peak sensitivity"

)



# value labels

for i in range(len(x)):


    ax2.text(

        x[i],

        mean_values[i]+0.05,

        f"{mean_values[i]:.2f}",

        ha="center",

        fontsize=8

    )


    ax2.text(

        x[i],

        peak_values[i]+0.25,

        f"DAP {peak_days[i]}",

        ha="center",

        fontsize=8

    )



ax2.set_xticks(x)


ax2.set_xticklabels(

    summary_df["Stage"],

    fontsize=9

)



ax2.set_ylabel(

    r"Sensitivity ($|\partial \hat{Y}/\partial I_t|$)"

)



ax2.set_xlabel(

    "Growth stages"

)



ax2.set_title(

    "(b) Growth-stage irrigation sensitivity",

    pad=4,

    fontsize=10

)



ax2.grid(

    axis="y",

    linestyle="--",

    alpha=0.3

)



ax2.spines["top"].set_visible(False)

ax2.spines["right"].set_visible(False)



ax2.set_ylim(0,4.8)

ax2.legend(

    frameon=False,

    loc="upper right", bbox_to_anchor=(1,1.08)

)



# ==================================================
# Save
# ==================================================

plt.subplots_adjust(bottom=0.16)

plt.savefig(

    OUT_PNG,

    dpi=600,

    bbox_inches="tight"

)



plt.subplots_adjust(bottom=0.16)

plt.savefig(

    OUT_PDF,

    bbox_inches="tight"

)



print("Finished")
print(OUT_PNG)
print(OUT_PDF)
print(OUT_CSV)
