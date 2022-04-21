use std::cmp;
use std::collections::HashMap;
use std::str::Chars;
use std::{str::FromStr, vec};

use inputs::{
    BINGO_BOARDS, CONSUPTION, CRAB_POSITIONS, DEPTHS, FISHES, MOVES, SEGMENT_MESSAGES, VECTORS, SMOKE_BASIN, CHUNKS
};
use log::debug;
mod inputs;

fn main() {
    env_logger::init();
    // day01_1();
    // day01_2();
    // day02_1();
    // day02_2();
    // day03_1();
    // day03_2();
    // day04_1();
    //day04_2();
    //day05_1();
    //day05_2();
    //day06_1();
    //day07_2();
    //day08_1();
    //day09_1();
    //day10_1();
    day10_2();
}


fn day01_1() -> i32 {
    let mut counter = 0;

    for i in 1..DEPTHS.len() {
        if DEPTHS[i] > DEPTHS[i - 1] {
            counter += 1;
        }
    }

    println!("Result of day1 {}", counter);
    counter
}

fn day01_2() {
    let mut windowedDepths = [0usize; 2000];

    for i in 2..DEPTHS.len() {
        windowedDepths[i - 2] += DEPTHS[i];
        windowedDepths[i - 2] += DEPTHS[i - 1];
        windowedDepths[i - 2] += DEPTHS[i - 2];
    }
    //println!("{:?}", windowedDepths);

    let mut counter = 0;
    for i in 0..windowedDepths.len() - 1 {
        if windowedDepths[i] < windowedDepths[i + 1] {
            counter += 1;
        }
    }

    println!("DEPTHS incresed {} times", counter)
}

fn day02_1() {
    let mut horizontal_pos: usize = 0;
    let mut vertical_pos: usize = 0;
    let movements: Vec<&str> = MOVES.split("\n").collect();

    for ele in movements {
        let current: Vec<_> = ele.split(" ").collect();

        let direction = current[0];
        let speed: usize = current[1].parse::<usize>().unwrap();

        match direction {
            "forward" => horizontal_pos += speed,
            "down" => vertical_pos += speed,
            "up" => vertical_pos -= speed,
            _ => unreachable!(),
        }
    }

    println!(
        "Position {}; {} => {}",
        horizontal_pos,
        vertical_pos,
        horizontal_pos * vertical_pos
    )
}

fn day02_2() {
    let mut horizontal_pos: usize = 0;
    let mut depth: usize = 0;
    let mut aim: usize = 0;
    let movements: Vec<&str> = MOVES.split("\n").collect();

    for ele in movements {
        let current: Vec<_> = ele.split(" ").collect();

        let direction = current[0];
        let speed: usize = current[1].parse::<usize>().unwrap();

        match direction {
            "forward" => {
                horizontal_pos += speed;
                depth += speed * aim;
            }
            "down" => aim += speed,
            "up" => aim -= speed,
            _ => unreachable!(),
        }
    }

    println!(
        "Position {}; {} => {}",
        horizontal_pos,
        depth,
        horizontal_pos * depth
    )
}

fn day03_1() {
    let len = CONSUPTION.len();

    let mut values = [0; 12];

    for ele in CONSUPTION {
        for i in 0usize..12 {
            let flag = 2usize.pow(i as u32);

            let val = (ele & flag) / flag;

            //println!("val {}, flag {}", val, flag);

            values[11 - i] += val;
        }
        //println!("ele {}, values {:?}", ele, values);
    }

    println!("Calculated values : {:?}", values);
    let mut gamma = 0usize;

    for (index, value) in values.iter().enumerate() {
        if *value > len / 2 {
            gamma += 2usize.pow((11 - index) as u32);
        }
    }

    println!("Gamma is {}", gamma);
    println!("Epsilon is {}", 4095 - gamma);
    println!("Res is {}", (4095 - gamma) * gamma);
}

fn day03_2() {
    let mut flags = [0usize; 12];
    for i in 0usize..12 {
        let flag = 2usize.pow(11 - i as u32);
        flags[i] = flag;
    }

    println!("Flags : {:?}", flags);

    let mut oxygen = 0usize;
    let mut co2 = 0usize;

    let mut current_left: Vec<usize> = CONSUPTION.into();
    let mut with_one = vec![];
    let mut with_zero = vec![];
    for flag in flags {
        with_one = vec![];
        with_zero = vec![];
        let len = current_left.len();
        for ele in current_left {
            let val = (ele & flag) / flag;

            if val == 1usize {
                with_one.push(ele);
            } else {
                with_zero.push(ele);
            }
        }
        // println!("with_one : {:?}", with_one.len());
        // println!("with_zero : {:?}", with_zero.len());

        if with_one.len() >= with_zero.len() {
            current_left = with_one;
        } else {
            current_left = with_zero;
        }

        if current_left.len() == 1 {
            oxygen = current_left[0];
            println!("This is the last one {}", oxygen);
            break;
        }
    }

    let mut current_left: Vec<usize> = CONSUPTION.into();
    let mut with_one = vec![];
    let mut with_zero = vec![];
    for flag in flags {
        with_one = vec![];
        with_zero = vec![];
        let len = current_left.len();
        for ele in current_left {
            let val = (ele & flag) / flag;

            if val == 1usize {
                with_one.push(ele);
            } else {
                with_zero.push(ele);
            }
        }
        // println!("with_one : {:?}", with_one.len());
        // println!("with_zero : {:?}", with_zero.len());

        if with_one.len() < with_zero.len() {
            current_left = with_one;
        } else {
            current_left = with_zero;
        }

        if current_left.len() == 1 {
            co2 = current_left[0];
            println!("This is the last one {}", co2);
            break;
        }
    }

    println!("Result is  {}", co2 * oxygen);
}

#[derive(Debug)]
struct Board {
    pub values: Vec<Number>,
}

impl Board {
    pub fn new(values: &str) -> Self {
        let values = values.replacen("\n", " ", 5);
        let values: Vec<&str> = values.split(" ").filter(|s| *s != "").collect();
        assert_eq!(values.len(), 25);
        let numbers: Vec<Number> = values
            .iter()
            .map(|s| Number::Unmarked(<i32 as FromStr>::from_str(s).unwrap()))
            .collect();
        Self { values: numbers }
    }

    pub fn mark(&mut self, val: i32) {
        for elem in &mut self.values {
            if let Number::Unmarked(current) = elem {
                if val == *current {
                    *elem = Number::Marked(val);
                }
            }
        }
    }

    pub fn check(&self) -> bool {
        let mut val = true;
        for j in 0..5 {
            val = true;
            for i in 0..5 {
                let index = i + j * 5;
                val &= self.values[index].checked();
            }

            if val {
                return val;
            }
        }

        for i in 0..5 {
            val = true;
            for j in 0..5 {
                let index = i + j * 5;
                val &= self.values[index].checked();
            }

            if val {
                return val;
            }
        }

        false
    }

    pub fn count_unchecked(&self) -> i32 {
        let mut value = 0;

        for elem in &self.values {
            match elem {
                Number::Unmarked(val) => value += val,
                _ => value += 0,
            }
        }

        value
    }
}

#[derive(Debug)]
enum Number {
    Marked(i32),
    Unmarked(i32),
}

impl Number {
    pub fn checked(&self) -> bool {
        match self {
            Number::Marked(_) => true,
            Number::Unmarked(_) => false,
        }
    }
}

fn day04_1() {
    let bingo = [
        4, 77, 78, 12, 91, 82, 48, 59, 28, 26, 34, 10, 71, 89, 54, 63, 66, 75, 15, 22, 39, 55, 83,
        47, 81, 74, 2, 46, 25, 98, 29, 21, 85, 96, 3, 16, 60, 31, 99, 86, 52, 17, 69, 27, 73, 49,
        95, 35, 9, 53, 64, 88, 37, 72, 92, 70, 5, 65, 79, 61, 38, 14, 7, 44, 43, 8, 42, 45, 23, 41,
        57, 80, 51, 90, 84, 11, 93, 40, 50, 33, 56, 67, 68, 32, 6, 94, 97, 13, 87, 30, 18, 76, 36,
        24, 19, 20, 1, 0, 58, 62,
    ];

    let mut boards = BINGO_BOARDS.map(|s| Board::new(s));

    for value in bingo {
        for board in &mut boards {
            board.mark(value);

            if board.check() {
                //println!("One board is all checked ! {:#?}", board);
                let res = board.count_unchecked() * value;
                println!("Result is ! {:#?}", res);

                return;
            }
        }
    }

    //println!("Board are like this : {:#?}", boards[0]);
}

fn day04_2() {
    let bingo = [
        4, 77, 78, 12, 91, 82, 48, 59, 28, 26, 34, 10, 71, 89, 54, 63, 66, 75, 15, 22, 39, 55, 83,
        47, 81, 74, 2, 46, 25, 98, 29, 21, 85, 96, 3, 16, 60, 31, 99, 86, 52, 17, 69, 27, 73, 49,
        95, 35, 9, 53, 64, 88, 37, 72, 92, 70, 5, 65, 79, 61, 38, 14, 7, 44, 43, 8, 42, 45, 23, 41,
        57, 80, 51, 90, 84, 11, 93, 40, 50, 33, 56, 67, 68, 32, 6, 94, 97, 13, 87, 30, 18, 76, 36,
        24, 19, 20, 1, 0, 58, 62,
    ];

    let mut boards: Vec<Board> = BINGO_BOARDS
        .map(|s| Board::new(s))
        .into_iter()
        .collect::<Vec<Board>>();

    for value in bingo {
        for board in &mut boards {
            board.mark(value);
        }

        if boards.len() == 1 {
            let last_board = &boards[0];
            if last_board.check() {
                println!("Last board is : {}", last_board.count_unchecked() * value);
                return;
            }
        }

        boards = boards.into_iter().filter(|b| !b.check()).collect();
    }
}

#[derive(Debug)]
struct Vector(i32, i32, i32, i32);

impl Vector {
    pub fn new(string: &str) -> Self {
        let mut string = string.replace(" -> ", ",");
        let values: Vec<i32> = string
            .split(",")
            .into_iter()
            .map(|s| <i32 as FromStr>::from_str(s).unwrap())
            .collect();

        Self(values[0], values[1], values[2], values[3])
    }

    pub fn getPoints(&self) -> Option<Vec<(i32, i32)>> {
        let mut points = vec![];

        if self.0 == self.2 {
            for i in cmp::min(self.1, self.3)..cmp::max(self.1, self.3) + 1 {
                points.push((self.0, i));
            }
        } else if self.1 == self.3 {
            for i in cmp::min(self.0, self.2)..cmp::max(self.0, self.2) + 1 {
                points.push((i, self.1));
            }
        } else {
            let mut current_x = self.0;
            let mut current_y = self.1;
            let lambda_x = match self.0 - self.2 {
                0 => 0,
                i32::MIN..=-1i32 => 1,
                1i32..=i32::MAX => -1,
            };

            let lambda_y = match self.1 - self.3 {
                0 => 0,
                i32::MIN..=-1i32 => 1,
                1i32..=i32::MAX => -1,
            };

            while current_x != self.2 && current_y != self.3 {
                points.push((current_x, current_y));
                current_x += lambda_x;
                current_y += lambda_y;
            }
            points.push((self.2, self.3));
        }

        Some(points)
    }
}

fn day05_1() {
    let vectors: Vec<Vector> = VECTORS.into_iter().map(|s| Vector::new(s)).collect();

    let mut board = [[0; 999]; 999];

    for vector in vectors {
        let points = vector.getPoints();
        if let Some(points) = points {
            for (x, y) in points {
                board[x as usize][y as usize] += 1;
            }
        }
    }

    let mut res = 0;
    for i in 0..999 {
        for j in 0..999 {
            if board[i][j] > 1 {
                res += 1;
            }
        }
    }

    println!("Result is {}", res);
}

fn day05_2() {
    let vectors: Vec<Vector> = VECTORS.into_iter().map(|s| Vector::new(s)).collect();

    let mut board = [[0; 999]; 999];

    for vector in vectors {
        let points = vector.getPoints();
        if let Some(points) = points {
            for (x, y) in points {
                board[x as usize][y as usize] += 1;
            }
        }
    }

    let mut res = 0;
    for i in 0..999 {
        for j in 0..999 {
            if board[i][j] > 1 {
                res += 1;
            }
        }
    }

    println!("Result is {}", res);
}

#[derive(Debug)]
struct Fishes(i128, i128, i128, i128, i128, i128, i128, i128, i128);

impl Fishes {
    pub fn new() -> Self {
        Self(0, 0, 0, 0, 0, 0, 0, 0, 0)
    }

    pub fn add(&mut self, i: i32) {
        match i {
            0 => self.0 += 1,
            1 => self.1 += 1,
            2 => self.2 += 1,
            3 => self.3 += 1,
            4 => self.4 += 1,
            5 => self.5 += 1,
            6 => self.6 += 1,
            7 => self.7 += 1,
            8 => self.8 += 1,
            _ => unreachable!(),
        };
    }

    pub fn decrement(&mut self) {
        let multiplating = self.0;
        self.0 = self.1;
        self.1 = self.2;
        self.2 = self.3;
        self.3 = self.4;
        self.4 = self.5;
        self.5 = self.6;
        self.6 = self.7 + multiplating;
        self.7 = self.8;
        self.8 = multiplating;
    }

    pub fn total(&self) -> i128 {
        self.0 + self.1 + self.2 + self.3 + self.4 + self.5 + self.6 + self.7 + self.8
    }
}

fn day06_1() {
    let mut fishes = Fishes::new();
    for ele in FISHES {
        fishes.add(ele);
    }
    println!("School is : {:?}", fishes);

    for i in 0..80 {
        fishes.decrement();
        println!("School is : {:?}", fishes);
    }

    println!("Number of fishes is : {:?}", fishes.total());
}

fn day06_2() {
    let mut fishes = Fishes::new();
    for ele in FISHES {
        fishes.add(ele);
    }
    println!("School is : {:?}", fishes);

    for i in 0..256 {
        fishes.decrement();
        println!("School is : {:?}", fishes);
    }

    println!("Number of fishes is : {:?}", fishes.total());
}

fn day07_2() {
    let crabs = CRAB_POSITIONS;

    let min_pos = crabs.clone().into_iter().min().unwrap();
    let max_pos = crabs.clone().into_iter().max().unwrap();

    println!("Min {}, Max {}", min_pos, max_pos);
    let mut current_fuel = i128::MAX;
    let mut res_pos = 0;
    for target in min_pos..max_pos + 1 {
        let fueld_used: i128 = crabs
            .clone()
            .into_iter()
            .map(|p| if p > target { p - target } else { target - p })
            .map(|n| (n * (n + 1)) / 2)
            .sum();

        println!("Testing on {} for {} fuel", target, fueld_used);

        if fueld_used < current_fuel {
            current_fuel = fueld_used;
            res_pos = target;
        }
    }

    println!("Every one on {} for {} fuel", res_pos, current_fuel)
}

#[derive(Debug)]
struct Message {
    pub patterns: [String; 10],
    pub numbers: Vec<String>,
    pub found: Vec<Option<String>>,
}

impl Message {
    pub fn new(string: &str) -> Self {
        let mut splits = string.split(" | ");

        let patterns = splits.next().unwrap();
        let patterns: Vec<String> = patterns
            .split(" ")
            .into_iter()
            .map(|s| s.to_owned())
            .map(|s| sort(s))
            .collect();
        let patterns: [String; 10] = patterns.try_into().unwrap();

        let numbers = splits.next().unwrap();
        let numbers: Vec<String> = numbers
            .split(" ")
            .into_iter()
            .map(|s| s.to_owned())
            .map(|s| sort(s))
            .collect();

        Self {
            patterns,
            numbers,
            found: vec![None; 10],
        }
    }

    pub fn get_number(&self) -> i32 {
        let mut res = 0;
        for (index, string) in self.numbers.iter().enumerate() {
            let temp_res = self.found.iter().position(|e| {
                if let Some(v) = e {
                    return *v == *string;
                }
                return false;
            });
            if temp_res.is_none() {
                println!("{:?}, {}", self, string);
            }

            let temp_res = temp_res.unwrap();

            res += (temp_res as i32) * 10i32.pow((3 - index) as u32);
        }

        res
    }

    pub fn get(&mut self, i: i32) -> String {
        if let Some(str) = self.found.get(i as usize).unwrap() {
            return str.to_string();
        }
        debug!("Getting i: {}", i);

        let res: String = match i {
            0 => {
                let six = self.get(6);
                let nine = self.get(9);
                self.patterns
                    .clone()
                    .into_iter()
                    .filter(|s| s.len() == 6)
                    .filter(|s| *s != six)
                    .filter(|s| *s != nine)
                    .next()
                    .unwrap()
            }
            1 => self
                .patterns
                .clone()
                .into_iter()
                .filter(|s| s.len() == 2)
                .next()
                .unwrap(),
            2 => {
                let three = self.get(3);
                let five = self.get(5);
                self.patterns
                    .clone()
                    .into_iter()
                    .filter(|s| s.len() == 5)
                    .filter(|s| *s != three && *s != five)
                    .next()
                    .unwrap()
            }
            3 => {
                let one = self.get(1);
                let mut one = one.chars();
                let first_segment = one.next().unwrap();
                let second_segment = one.next().unwrap();
                self.patterns
                    .clone()
                    .into_iter()
                    .filter(|s| s.len() == 5)
                    .filter(|s| s.contains(first_segment) && s.contains(second_segment))
                    .next()
                    .unwrap()
            }
            4 => self
                .patterns
                .clone()
                .into_iter()
                .filter(|s| s.len() == 4)
                .next()
                .unwrap(),
            5 => {
                let six = self.get(6);
                self.patterns
                    .clone()
                    .into_iter()
                    .filter(|s| s.len() == 5)
                    .filter(|s| uncommon_char(&six, &s) == 1)
                    .next()
                    .unwrap()
            }
            6 => {
                let one = self.get(1);
                let one_chars = one.chars();
                let six_length: Vec<String> = self
                    .patterns
                    .clone()
                    .into_iter()
                    .filter(|s| s.len() == 6)
                    .collect();

                let matching: Vec<String> = six_length
                    .into_iter()
                    .filter(|s| !contains(s, one_chars.clone()))
                    .collect();

                matching.get(0).unwrap().to_string()
            }
            7 => self
                .patterns
                .clone()
                .into_iter()
                .filter(|s| s.len() == 3)
                .next()
                .unwrap(),
            8 => self
                .patterns
                .clone()
                .into_iter()
                .filter(|s| s.len() == 7)
                .next()
                .unwrap(),
            9 => {
                let four = self.get(4);
                let mut four_chars = four.chars();
                self.patterns
                    .clone()
                    .into_iter()
                    .filter(|s| s.len() == 6)
                    .filter(|s| contains(s, four_chars.clone()))
                    .next()
                    .unwrap()
            }
            _ => unreachable!(),
        };

        debug!("Found: {} for {}", res, i);

        self.found[i as usize] = Some(res.to_string());

        return res.to_string();
    }
}

fn sort(string: String) -> String {
    let mut chars: Vec<char> = string.chars().collect();
    chars.sort_by(|a, b| a.cmp(b));
    String::from_iter(chars)
}

fn uncommon_char(str1: &String, str2: &String) -> i32 {
    let mut count = 0i32;

    for char in str1.chars() {
        if !str2.contains(char) {
            count += 1;
        }
    }
    count
}


fn contains(s: &String, chars: Chars) -> bool {
    let mut res = true;
    for char in chars {
        res = res && s.contains(char);
    }
    res
}

fn day08_1() {
    let mut messages: Vec<Message> = SEGMENT_MESSAGES
        .into_iter()
        .map(|s| Message::new(s))
        .collect();

    let mut res = 0i128;
    let mut index = 0;

    for mut message in messages {
        debug!("doing : {:?}", message);
        for i in 0..10 {
            message.get(i);
        }
        index += 1;
        res += message.get_number() as i128;

        println!("index {} done", index);
    }

    println!("Result is {}", res);
}

fn day09_1() {

    let height = SMOKE_BASIN.len() as isize;
    let width = SMOKE_BASIN[0].len()as isize;

    let mut smoke: Vec<Vec<u32>> = SMOKE_BASIN.iter()
        .map(|s| s.chars())
        .map(|c| c.map(|c| c.to_digit(10).unwrap()).collect())
        .collect();

    debug!("Smoke is {:?}", smoke);

    let mut low_points: Vec<(isize, isize)> = vec![];

    for i in 0..height {
        for j in 0..width {
            debug!("Smoke of i {:?}", smoke[i as usize]);

            let location = get_location(&mut smoke, i, j);
            debug!("Location of ({},{}) is {}", i, j, location);

            let adjacents: Vec<(isize, isize)> = get_adjacent(i,j, height, width);
            debug!("adjacent of ({},{}) are {:?}", i, j, adjacents);
            let min_adjacdent_locations: u32 = adjacents.iter().map(|(i,j)| get_location(&mut smoke, *i, *j)).min().unwrap();

            if location < min_adjacdent_locations { 
                low_points.push((i,j));
             }
        }
    }


    let mut points: Vec<Vec<Option<Point>>> = vec![];
    for  x in 0isize..100 {
        let mut tmp_points = vec![];
        for  y in 0isize..100 {
            let val = smoke[x as usize][y as usize];
            tmp_points.push(Point::new(val, (x,y)))
        }
        points.push(tmp_points);
    }

    let mut map = DepthMap { points, low_points, size: vec![] };
    map.compute_all();
    print!("{:?}", map.size);
    print!("{:?}", map.low_points[0]);
}

#[derive(Debug)]
struct DepthMap {
    pub points: Vec<Vec<Option<Point>>>,
    pub low_points: Vec<(isize, isize)>,
    pub size: Vec<isize>,
}

impl DepthMap {

    fn compute_all(&mut self) {
        for (x, y) in &self.low_points {
            let mut size = 0;
            let mut neighboors = vec![(*x, *y)];
            let mut growing= true;
            while growing {
                let mut next_neighboors: Vec<(isize, isize)> = vec![];
                growing = false;
                for neighboor in neighboors {
                    let point = self.points[neighboor.0 as usize][neighboor.1 as usize].as_mut();
                    if let Some(point) = point {
                        if !point.visited {
                            growing = true;
                            size += 1;
                            point.visited = true;
                            let other = &mut get_adjacent(point.coord.0, point.coord.1, 100,100);
                            next_neighboors.append(other);
                            //println!("({},{}) {} Visiting", neighboor.0, neighboor.1, point.val);
                        }else {
                            //println!("({},{}) {} Already visited", neighboor.0, neighboor.1, point.val);
                        }
                    }else {
                            //println!("({},{}) 9 is None", neighboor.0, neighboor.1);
                    }
                }
                neighboors = next_neighboors;

            }
            self.size.push(size);
        }
        // for points in &self.points {
        //     for point in points {
        //         if let Some(point) = point {
        //             print!("1 ");
        //         }else {
        //             print!("0 ");
        //         }
        //     }
        //     println!()
        // }
        //println!("{:?}", self.points);
        self.size.sort();
    }

}

#[derive(Clone, Debug)]
struct Point {
    pub val: u32,
    pub visited: bool,
    pub coord:(isize, isize),
}

impl Point {
    fn new(val: u32, coord: (isize, isize)) -> Option<Self> {
        if val == 9 {
            return None;
        }
        Some(Self {
            val,
            visited: false,
            coord
        })
    }
}

fn get_location(smoke: &mut Vec<Vec<u32>>, i: isize, j: isize) -> u32 {
    smoke[i as usize][j as usize]
}

fn get_adjacent(i: isize, j: isize, height: isize, width: isize) -> Vec<(isize, isize)> {
    let res = vec![
        (i-1, j), (i, j-1), (i+1, j), (i, j+1)
    ];

    return res.into_iter()
        .filter(|(i,j)| i>= &0 && j >= &0 && i < &height && j < &width)
    .collect();
}



fn day10_1() {
    let mut res : Vec<Option<i32>> = vec![];
    for lines in CHUNKS {
        res.push(analyse_line(lines));
    }

    let value: i32 = res.iter().filter_map(|s| *s).sum();
    println!("Value is {}", value);
}


fn day10_2() {
    let mut scores = vec![];
    for lines in CHUNKS {
        let resulst = analyse_line_2(lines);

        if let Some(mut result) = resulst {
            let mut values = 0i128;
            result.reverse();

            let my_string: String = String::from_iter(result.clone());
            println!("Line is  {:?}", lines);
            println!("Calculating for {:?}", my_string);

            for c in result {
                let add_this = match c {
                    ')' => 1, 
                    ']' => 2, 
                    '}' => 3,
                    '>' => 4, 
                    _ => unreachable!()
               };

               values = values * 5 + add_this;
            }
            println!("Found {:?}", values);


            scores.push(values);
        }
    }

    scores.sort();
    println!("{:?}", scores);
    println!("Size is {}, middle is {}", scores.len(), scores.len() / 2);
    println!("score is {}", scores[scores.len() / 2]);

}



fn analyse_line(lines: &str) -> Option<i32> {
    let chars = lines.chars();
    let mut expected: Vec<char> = vec![];
    for char in chars {
        let char_type: Type = char.into();

        match char_type {
            Type::STARTING(c) => {
                expected.push(matching_of(c));
            },
            Type::ENDING(found) => {
                let expecting = expected.pop();
                match expecting {
                    Some(c) => {
                        if c != found {
                            return Some(get_points(found));
                        }
                    },
                    None => {
                        return Some(get_points(found));
                    },
                }
            },
        }
    }

    return None;
}

fn analyse_line_2(lines: &str) -> Option<Vec<char>> {
    let chars = lines.chars();
    let mut expected: Vec<char> = vec![];
    for char in chars {
        let char_type: Type = char.into();

        match char_type {
            Type::STARTING(c) => {
                expected.push(matching_of(c));
            },
            Type::ENDING(found) => {
                let expecting = expected.pop();
                match expecting {
                    Some(c) => {
                        if c != found {
                            return None;
                        }
                    },
                    None => {
                        return None;
                    },
                }
            },
        }
    }

    return Some(expected);
}

fn matching_of(c: char) -> char {
    match c {
        '<' => '>', 
        '(' => ')', 
        '[' => ']', 
        '{' => '}',
        _ => unreachable!()
    }
}

fn get_points(c: char) -> i32 {
    match c {
         ')' => 3, 
         ']' => 57, 
         '}' => 1197,
         '>' => 25137, 
         _ => unreachable!()
    }
}

enum Type {
    STARTING(char),
    ENDING(char),
}

impl From<char> for Type {
    fn from(c: char) -> Self { 
        match c {
            '<' | '(' | '[' | '{' => Type::STARTING(c),
            '>' | ')' | ']' | '}' => Type::ENDING(c),
            _ => unreachable!()
        }
    }
}