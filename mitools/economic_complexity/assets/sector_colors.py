from pandas import Interval

SECTOR_BINS = {
    "HS2 ID": {
        "color": {
            Interval(left=101, right=105, closed="right"): "#F57373",
            Interval(left=206, right=214, closed="right"): "#AAD75A",
            Interval(left=315, right=315, closed="right"): "#D7B55A",
            Interval(left=416, right=424, closed="right"): "#FAF51B",
            Interval(left=525, right=527, closed="right"): "#643F0D",
            Interval(left=628, right=638, closed="right"): "#EA14B0",
            Interval(left=739, right=740, closed="right"): "#854CEB",
            Interval(left=841, right=843, closed="right"): "#B87B7B",
            Interval(left=944, right=946, closed="right"): "#204F0B",
            Interval(left=1047, right=1049, closed="right"): "#F8F0A4",
            Interval(left=1150, right=1163, closed="right"): "#00CC1E",
            Interval(left=1264, right=1267, closed="right"): "#00FF25",
            Interval(left=1368, right=1370, closed="right"): "#AB9E97",
            Interval(left=1471, right=1471, closed="right"): "#76E5E7",
            Interval(left=1572, right=1583, closed="right"): "#4E6D6E",
            Interval(left=1684, right=1685, closed="right"): "#314480",
            Interval(left=1786, right=1789, closed="right"): "#85F6FF",
            Interval(left=1890, right=1892, closed="right"): "#3F004C",
            Interval(left=1993, right=1993, closed="right"): "#B3B3B3",
            Interval(left=2094, right=2096, closed="right"): "#614906",
            Interval(left=2197, right=2197, closed="right"): "#B584C4",
            Interval(left=2299, right=2299, closed="right"): "#000000",
        },
        "names": [
            "Animal Products",
            "Vegetable Products",
            "Animal and Vegetable Bi-Products",
            "Foodstuffs",
            "Mineral Products",
            "Chemical Products",
            "Plastics and Rubbers",
            "Animal Hides",
            "Wood Products",
            "Paper Goods",
            "Textiles",
            "Footwear and Headwear",
            "Stone And Glass",
            "Precious Metals",
            "Metals",
            "Machines",
            "Transportation",
            "Instruments",
            "Weapons",
            "Miscellaneous",
            "Arts and Antiques",
            "Unspecified",
        ],
    },
    "HS4 ID": {
        "color": {
            Interval(left=10101, right=10511, closed="right"): "#F57373",
            Interval(left=20601, right=21404, closed="right"): "#AAD75A",
            Interval(left=31501, right=31522, closed="right"): "#D7B55A",
            Interval(left=41601, right=42403, closed="right"): "#FAF51B",
            Interval(left=52501, right=52716, closed="right"): "#643F0D",
            Interval(left=62801, right=63826, closed="right"): "#EA14B0",
            Interval(left=73901, right=74017, closed="right"): "#854CEB",
            Interval(left=84101, right=84304, closed="right"): "#B87B7B",
            Interval(left=94401, right=94602, closed="right"): "#204F0B",
            Interval(left=104701, right=104911, closed="right"): "#F8F0A4",
            Interval(left=115001, right=116310, closed="right"): "#00CC1E",
            Interval(left=126401, right=126703, closed="right"): "#00FF25",
            Interval(left=136801, right=137020, closed="right"): "#AB9E97",
            Interval(left=147101, right=147118, closed="right"): "#76E5E7",
            Interval(left=157201, right=158311, closed="right"): "#4E6D6E",
            Interval(left=168401, right=168548, closed="right"): "#314480",
            Interval(left=178601, right=178908, closed="right"): "#85F6FF",
            Interval(left=189001, right=189209, closed="right"): "#3F004C",
            Interval(left=199301, right=199307, closed="right"): "#B3B3B3",
            Interval(left=209401, right=209619, closed="right"): "#614906",
            Interval(left=219701, right=219706, closed="right"): "#B584C4",
            Interval(left=229901, right=229999, closed="right"): "#000000",
        },
        "names": [
            "Animal Products",
            "Vegetable Products",
            "Animal and Vegetable Bi-Products",
            "Foodstuffs",
            "Mineral Products",
            "Chemical Products",
            "Plastics and Rubbers",
            "Animal Hides",
            "Wood Products",
            "Paper Goods",
            "Textiles",
            "Footwear and Headwear",
            "Stone And Glass",
            "Precious Metals",
            "Metals",
            "Machines",
            "Transportation",
            "Instruments",
            "Weapons",
            "Miscellaneous",
            "Arts and Antiques",
            "Unspecified",
        ],
    },
    "HS6 ID": {
        "color": {
            Interval(left=1010100, right=1051199, closed="right"): "#F57373",
            Interval(left=2060100, right=2140499, closed="right"): "#AAD75A",
            Interval(left=3150100, right=3152299, closed="right"): "#D7B55A",
            Interval(left=4160100, right=4240399, closed="right"): "#FAF51B",
            Interval(left=5250100, right=5271699, closed="right"): "#643F0D",
            Interval(left=6280100, right=6382699, closed="right"): "#EA14B0",
            Interval(left=7390100, right=7401799, closed="right"): "#854CEB",
            Interval(left=8410100, right=8430499, closed="right"): "#B87B7B",
            Interval(left=9440100, right=9460299, closed="right"): "#204F0B",
            Interval(left=10470100, right=10491199, closed="right"): "#F8F0A4",
            Interval(left=11500100, right=11631099, closed="right"): "#00CC1E",
            Interval(left=12640100, right=12670399, closed="right"): "#00FF25",
            Interval(left=13680100, right=13702099, closed="right"): "#AB9E97",
            Interval(left=14710100, right=14711899, closed="right"): "#76E5E7",
            Interval(left=15720100, right=15831199, closed="right"): "#4E6D6E",
            Interval(left=16840100, right=16854899, closed="right"): "#314480",
            Interval(left=17860100, right=17890899, closed="right"): "#85F6FF",
            Interval(left=18900100, right=18920999, closed="right"): "#3F004C",
            Interval(left=19930100, right=19930799, closed="right"): "#B3B3B3",
            Interval(left=20940100, right=20961999, closed="right"): "#614906",
            Interval(left=21970100, right=21970699, closed="right"): "#B584C4",
            Interval(left=22990100, right=22999999, closed="right"): "#000000",
        },
        "names": [
            "Animal Products",
            "Vegetable Products",
            "Animal and Vegetable Bi-Products",
            "Foodstuffs",
            "Mineral Products",
            "Chemical Products",
            "Plastics and Rubbers",
            "Animal Hides",
            "Wood Products",
            "Paper Goods",
            "Textiles",
            "Footwear and Headwear",
            "Stone And Glass",
            "Precious Metals",
            "Metals",
            "Machines",
            "Transportation",
            "Instruments",
            "Weapons",
            "Miscellaneous",
            "Arts and Antiques",
            "Unspecified",
        ],
    },
    "sitc_product_code": {
        "color": {
            Interval(left=0, right=999, closed="right"): "#65D100",
            Interval(left=1000, right=1999, closed="right"): "#FFE400",
            Interval(left=2000, right=2999, closed="right"): "#DE1313",
            Interval(left=3000, right=3999, closed="right"): "#89422F",
            Interval(left=4000, right=4999, closed="right"): "#F7A5F4",
            Interval(left=5000, right=5999, closed="right"): "#FF199A",
            Interval(left=6000, right=6999, closed="right"): "#6219FF",
            Interval(left=7000, right=7999, closed="right"): "#6EABFF",
            Interval(left=8000, right=8999, closed="right"): "#27895E",
            Interval(left=9000, right=9999, closed="right"): "#000000",
        },
        "names": [
            "Food & Live Animals",
            "Beverages & Tobacco",
            "Raw Materials",
            "Mineral fuels, Lubricants & Related Materials",
            "Animal & Vegetable Oils, Fats & Waxes",
            "Chemicals",
            "Manufactured Goods by Material",
            "Machinery & Transport Equipment",
            "Miscellaneous Manufactured Articles",
            "Miscellaneous",
        ],
    },
    "Sector": {
        "color": {
            Interval(left=101, right=105, closed="right"): "#F57373",
            Interval(left=206, right=214, closed="right"): "#AAD75A",
            Interval(left=315, right=315, closed="right"): "#D7B55A",
            Interval(left=416, right=424, closed="right"): "#FAF51B",
            Interval(left=525, right=527, closed="right"): "#643F0D",
            Interval(left=628, right=638, closed="right"): "#EA14B0",
            Interval(left=739, right=740, closed="right"): "#854CEB",
            Interval(left=841, right=843, closed="right"): "#B87B7B",
            Interval(left=944, right=946, closed="right"): "#204F0B",
            Interval(left=1047, right=1049, closed="right"): "#F8F0A4",
            Interval(left=1150, right=1163, closed="right"): "#00CC1E",
            Interval(left=1264, right=1267, closed="right"): "#00FF25",
            Interval(left=1368, right=1370, closed="right"): "#AB9E97",
            Interval(left=1471, right=1471, closed="right"): "#76E5E7",
            Interval(left=1572, right=1583, closed="right"): "#4E6D6E",
            Interval(left=1684, right=1685, closed="right"): "#314480",
            Interval(left=1786, right=1789, closed="right"): "#85F6FF",
            Interval(left=1890, right=1892, closed="right"): "#3F004C",
            Interval(left=1993, right=1993, closed="right"): "#B3B3B3",
            Interval(left=2094, right=2096, closed="right"): "#614906",
            Interval(left=2197, right=2197, closed="right"): "#B584C4",
            Interval(left=2298, right=2298, closed="right"): "#000000",
            Interval(left=9999, right=99105, closed="right"): "#643F0D",
            Interval(left=99106, right=99107, closed="right"): "#242c57",
            Interval(left=99108, right=99110, closed="right"): "#85F6FF",
            Interval(left=99111, right=99111, closed="right"): "#a949fc",
            Interval(left=99113, right=99115, closed="right"): "#ff036c",
            Interval(left=99116, right=99116, closed="right"): "#3e3d40",
            Interval(left=99117, right=99119, closed="right"): "#ffe0ab",
            Interval(left=99120, right=99120, closed="right"): "#59dea2",
            Interval(left=99121, right=99121, closed="right"): "#ebadff",
            Interval(left=99122, right=99124, closed="right"): "#056ef7",
            Interval(left=99125, right=99125, closed="right"): "#b157cf",
            Interval(left=99126, right=99127, closed="right"): "#7391f5",
            Interval(left=99128, right=99128, closed="right"): "#4a5947",
            Interval(left=99129, right=99129, closed="right"): "#a3021a",
            Interval(left=99130, right=99130, closed="right"): "#a9db04",
            Interval(left=99131, right=99131, closed="right"): "#ff059f",
            Interval(left=99132, right=99132, closed="right"): "#addeff",
            Interval(left=99133, right=99133, closed="right"): "#0019bf",
            Interval(left=99134, right=99134, closed="right"): "#bf5900",
            Interval(left=99135, right=99135, closed="right"): "#fcb3eb",
        },
        "names": [
            "Animal Products",
            "Vegetable Products",
            "Animal and Vegetable Bi-Products",
            "Foodstuffs",
            "Mineral Products",
            "Chemical Products",
            "Plastics and Rubbers",
            "Animal Hides",
            "Wood Products",
            "Paper Goods",
            "Textiles",
            "Footwear and Headwear",
            "Stone And Glass",
            "Precious Metals",
            "Metals",
            "Machines",
            "Transportation",
            "Instruments",
            "Weapons",
            "Miscellaneous",
            "Arts and Antiques",
            "Unspecified",
            "Mineral Fuels, Oils, Distillation Products, etc",
            "Manufacturing & Maintenance",
            "Transport",
            "Postal & Courier Services",
            "Travel Goods",
            "Local Transport, Acommodation, Food-Serving Services",
            "Construction",
            "Insurance, Pension, Financial Services",
            "Real State",
            "Intellectual Property",
            "Information & Technology Services",
            "Research & Development",
            "Consulting & Engineering Services",
            "Waste Treatment",
            "Operational Leasing Services",
            "Trade & Business Services",
            "Audiovisual Services",
            "Health Services",
            "Education Services",
            "Heritage & Recreational Services",
            "Government Goods and Services",
        ],
    },
}

if __name__ == "__main__":
    pass