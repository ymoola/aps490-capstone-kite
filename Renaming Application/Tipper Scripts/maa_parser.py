import pandas as pd                    
from pathlib import Path

# -----------------------------
# Parse a single MAA sheet
# -----------------------------
def parse_maa_sheet(
    df_raw: pd.DataFrame,
    date: str,
    participant_id: str,
    shoeid: str
) -> pd.DataFrame:
    # Locate header row
    header_row = df_raw.index[df_raw.iloc[:, 0] == "Angle"][0]

    headers = df_raw.iloc[header_row].tolist()
    data = df_raw.iloc[header_row + 1:].reset_index(drop=True)
    data.columns = headers

    angle_col = "Angle"

    # Identify (File, U, D) blocks
    blocks = []
    i = 0
    while i < len(headers) - 2:
        if headers[i] == "File" and headers[i + 1] == "U" and headers[i + 2] == "D":
            blocks.append((i, i + 1, i + 2))
            i += 3
        else:
            i += 1

    records = []

    for _, row in data.iterrows():
        angle = row.get(angle_col)

        if pd.isna(angle):
            continue

        for file_idx, u_idx, d_idx in blocks:
            file_num = row.iloc[file_idx]

            if pd.isna(file_num):
                continue

            # Up
            u_val = row.iloc[u_idx]
            if u_val in [0, 1]:
                records.append({
                    "date": date,
                    "participant_id": participant_id,
                    "shoeid": shoeid,
                    "file": int(file_num),
                    "angle": int(angle),
                    "direction": "U",
                    "result": int(u_val)
                })

            # Down
            d_val = row.iloc[d_idx]
            if d_val in [0, 1]:
                records.append({
                    "date": date,
                    "participant_id": participant_id,
                    "shoeid": shoeid,
                    "file": int(file_num),
                    "angle": int(angle),
                    "direction": "D",
                    "result": int(d_val)
                })

    return pd.DataFrame(records)


# -----------------------------
# Process one participant file
# -----------------------------
def process_participant_file(
    file_path: Path,
    date: str
) -> pd.DataFrame:
    participant_id = file_path.stem
    xls = pd.ExcelFile(file_path)

    dfs = []

    for sheet_name in xls.sheet_names:
        if not sheet_name.startswith("iDAPT"):
            continue

        df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)

        df_sheet = parse_maa_sheet(
            df_raw=df_raw,
            date=date,
            participant_id=participant_id,
            shoeid=sheet_name
        )

        dfs.append(df_sheet)

    if dfs:
        return pd.concat(dfs, ignore_index=True)

    return pd.DataFrame(columns=[
        "date", "participant_id", "shoeid",
        "file", "angle", "direction", "result"
    ])


# -----------------------------
# Walk entire MAA directory
# -----------------------------
def process_maa_directory(
    maa_root: str,
    output_excel: str
):
    maa_root = Path(maa_root)
    all_data = []

    for date_dir in sorted(p for p in maa_root.iterdir() if p.is_dir()):
        date = date_dir.name

        for excel_file in date_dir.glob("*.xlsx"):
            df_participant = process_participant_file(
                file_path=excel_file,
                date=date
            )

            if not df_participant.empty:
                all_data.append(df_participant)

    if not all_data:
        raise RuntimeError("No valid MAA data found.")

    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.sort_values(by=["date", "participant_id", "shoeid", "file"],
    ascending=[True, True, True, True]
)

    final_df.to_excel(output_excel, index=False)

    return final_df


if __name__ == "__main__":
    process_maa_directory(
        maa_root="MAA",
        output_excel="maa_parsed.xlsx"
    )
